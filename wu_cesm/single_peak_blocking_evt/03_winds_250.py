#!/usr/bin/env python3
"""03 — the 250 hPa winds: external reference, the two pieces, and the closure.

    (a) V_rot_anom   Helmholtz rotational-wind anomaly straight from the archive `_rot.nc`.
                     EXTERNAL to the inversion — nothing in the PPVI produced it.
    (b) V_block      induced by `pv_anom_p`
    (c) V_eddy       induced by `pv_anom_e`
    (d) closure      (V_block + V_eddy) - V_rot_anom

V_tot IS NOT V_rot — THE TWO "TOTALS" ARE DIFFERENT QUANTITIES
--------------------------------------------------------------
This trips people up, so it is worth being blunt about:

  V_tot   the INVERSION's own total: the same upper-piece PV, inverted WITHOUT the mask.
          Entirely internal to the PPVI. Nothing observed about it.
  V_rot   the archive Helmholtz rotational-wind anomaly. Entirely EXTERNAL. Nothing in the
          inversion produced it.

They are not close: at 250 hPa over the plotted window, RMS 16.9 vs 15.3 m/s, corr 0.974 — the
upper piece alone OVER-produces the amplitude by 10% while matching the pattern well.

So the two "closure" numbers on this figure measure different things:

  V_block + V_eddy - V_tot   IDENTICALLY zero. `psi_e` is DEFINED as `psi_tot - psi_p` and
                             psi->wind is linear, so V_e := V_tot - V_p. This checks that the
                             SPLIT was done without an arithmetic slip and says nothing whatever
                             about whether V_tot resembles the atmosphere. Reported as a number,
                             never plotted — a panel of zeros is not a test.
  (V_block + V_eddy) - V_rot = V_tot - V_rot. THIS is panel (d), and this is where the lower and
                             surface pieces and the unbalanced flow live.

Panel (d) is NOT expected to vanish, and it has a FLOOR. `_residual_origin.py` measures the split:
inverting the lower and surface pieces too drops the residual from 36% to 27% of the reference
(full window), with the summed amplitude then matching V_rot to 0.5% at corr 0.964. The remaining
27% is exactly `sqrt(2(1-r))` for that correlation — it is not a missing piece and not a bias. It
is the unbalanced rotational flow, which nonlinear balance cannot represent, plus the fact that the
balance operator is NONLINEAR so three piece solutions do not sum to one full inversion. No amount
of extra iteration or extra pieces removes it. "Smaller (d) is better" is therefore wrong.

This is the same discipline the rest of this work has had to learn the hard way: internal
self-consistency can never establish that an iterative solver converged. Only an external reference
can, and `_rot.nc` is the only one available here.

SHADING IS THE PV, NOT THE WIND SPEED. Each panel shades the PV anomaly that induced its wind —
(b) over `pv_anom_p`, (c) over `pv_anom_e` — so "this wind came from this PV" is directly readable.
(a) and (d) both involve the total, so both show the total. Winds are vectors only, on one quiver
scale across all four panels, with a key: with speed no longer shaded, that key is the only
quantitative wind information on the figure. RMS values are printed to stdout and in the caption.

Reads `_cache.npz` written by 01.

Run:  micromamba run -n blocking python 03_winds_250.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _evt
import build_pvbudget_15deg as Bg

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_cache.npz"
GRID = 1.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lon-half", type=float, default=60.0, help="longitude half-width to plot")
    ap.add_argument("--skip", type=int, nargs=2, default=(3, 3), help="vector subsample (y, x)")
    args = ap.parse_args()

    if not CACHE.exists():
        raise SystemExit("no _cache.npz — run 01_pv_anom_sections.py first")
    z = np.load(CACHE, allow_pickle=True)
    meta, info = eval(str(z["meta"])), eval(str(z["info"]))
    icen = int(z["icen"])
    y = (z["band"] - float(meta["clat"]))[::-1]                 # ascending latitude
    x = np.arange(-icen, icen + 1) * GRID

    kw = _evt.WIND_HPA.index(250)                                # in u_p / u_e / u_tot (400..200)
    kr = [_evt.WU[i] for i in Bg.WAVG_IDX].index(250)            # in u_rot_anom (wavg: 300/250/200)
    fl = lambda a: a[::-1]                                       # N->S  ->  ascending

    F = {"ref":   (fl(z["u_rot_anom"][kr]), fl(z["v_rot_anom"][kr])),
         "block": (fl(z["u_p"][kw]),        fl(z["v_p"][kw])),
         "eddy":  (fl(z["u_e"][kw]),        fl(z["v_e"][kw]))}
    F["clos"] = (F["block"][0] + F["eddy"][0] - F["ref"][0],
                 F["block"][1] + F["eddy"][1] - F["ref"][1])
    SPD = {k: np.hypot(*v) for k, v in F.items()}
    # Shading is the PV that induced each wind, piece-matched: (b) shows V_block over the very PV
    # anomaly it was inverted from. (a) and (d) both involve the total, so they get the total.
    PV = {"ref":   fl(z["pv_anom"][_evt.WU.index(250)]),
          "block": fl(z["pv_anom_p"][_evt.WU.index(250)]),
          "eddy":  fl(z["pv_anom_e"][_evt.WU.index(250)]),
          "clos":  fl(z["pv_anom"][_evt.WU.index(250)])}

    # the identity check, as a number — see the docstring
    du = z["u_p"][kw] + z["u_e"][kw] - z["u_tot"][kw]
    dv = z["v_p"][kw] + z["v_e"][kw] - z["v_tot"][kw]
    ident = float(np.nanmax(np.hypot(du, dv)))

    ix = np.abs(x) <= args.lon_half
    xc, sub = x[ix], (slice(None, None, args.skip[0]), slice(None, None, args.skip[1]))

    pmax = float(np.nanpercentile(np.abs(PV["ref"][:, ix]), 99.5))
    lev = np.linspace(-pmax, pmax, 21)
    smax = float(np.nanpercentile(np.concatenate([SPD[k][:, ix].ravel()
                                                  for k in ("ref", "block", "eddy")]), 99.5))
    qs = smax * 7                                                # one quiver scale for all panels
    qref = 5 * round(smax / 2 / 5) or 5                          # a round number for the key

    titles = {"ref":   r"$\mathbf{(a)}$  $\mathbf{V}'_{rot}$  (Helmholtz, archive — EXTERNAL)",
              "block": r"$\mathbf{(b)}$  $\mathbf{V}'_{block}$  (from $q'_p$)",
              "eddy":  r"$\mathbf{(c)}$  $\mathbf{V}'_{eddy}$  (from $q'_e$)",
              "clos":  r"$\mathbf{(d)}$  $(\mathbf{V}'_{block}+\mathbf{V}'_{eddy})"
                       r"-\mathbf{V}'_{rot}$"}

    # The gap between the columns is NOT wspace — it is `aspect="equal"` refusing to fill a cell
    # that is the wrong shape. Each panel is 120 deg lon x 75 deg lat = 1.6:1, so the figure has to
    # be sized to match or matplotlib centres a 1.6:1 panel inside a 2.3:1 cell and the leftover
    # becomes whitespace. figW/figH ~ 1.45 with hspace 0.12 cuts the slack from 32% to ~7%.
    fig, axes = plt.subplots(2, 2, figsize=(13.1, 8.8), sharex=True, sharey=True)
    for ax, key in zip(axes.ravel(), ("ref", "block", "eddy", "clos")):
        u, v = F[key][0][:, ix], F[key][1][:, ix]
        cf = ax.contourf(xc, y, PV[key][:, ix], levels=lev, cmap="RdBu_r", extend="both")
        q = ax.quiver(xc[sub[1]], y[sub[0]], u[sub], v[sub], scale=qs, width=0.0030,
                      color="0.08")
        # With speed no longer shaded, the key is the ONLY quantitative wind information on the
        # figure. ONE key: all four panels share `qs`, and a key on the right column would sit
        # underneath the colourbar.
        if key == "ref":
            ax.add_patch(plt.Rectangle((0.615, 0.015), 0.375, 0.105, transform=ax.transAxes,
                                       fc="white", ec="0.55", lw=0.6, zorder=5))
            ax.quiverkey(q, 0.665, 0.068, qref, f"{qref:.0f} m s$^{{-1}}$  (all panels)",
                         labelpos="E", coordinates="axes", zorder=6,
                         fontproperties=dict(size=8.5))
        ax.plot(0, 0, "k+", ms=13, mew=2.2)
        ax.set_title(titles[key], fontsize=10.5, loc="left", x=0.01)
        ax.set_aspect("equal")
    for ax in axes[1]:
        ax.set_xlabel("relative longitude [deg]")
    for ax in axes[:, 0]:
        ax.set_ylabel("relative latitude [deg]")

    # the object outline at 250 hPa, on every panel — where the block PV actually is
    M250 = np.full_like(SPD["ref"], np.nan)
    M250[:] = fl(z["M"][_evt.OBJ_HPA.index(250)])
    for ax in axes.ravel():
        ax.contour(xc, y, M250[:, ix], levels=[0.5], colors="#00e5ff", linewidths=1.8)

    rms = lambda a: float(np.sqrt(np.nanmean(a[:, ix] ** 2)))
    fr = rms(SPD["clos"]) / rms(SPD["ref"])
    # V_tot vs V_rot — the EXTERNAL comparison, kept distinct from the internal identity above
    _ut, _vt = fl(z["u_tot"][kw])[:, ix], fl(z["v_tot"][kw])[:, ix]
    _xx = np.concatenate([_ut.ravel(), _vt.ravel()])
    _yy = np.concatenate([F["ref"][0][:, ix].ravel(), F["ref"][1][:, ix].ravel()])
    _m = np.isfinite(_xx) & np.isfinite(_yy)
    ctr = float(np.corrcoef(_xx[_m], _yy[_m])[0, 1])
    rtr = float(np.sqrt(np.nanmean(_ut ** 2 + _vt ** 2)) / rms(SPD["ref"]))
    fig.suptitle(
        f"03 · 250 hPa anomaly winds — m{meta['member']} "
        f"{meta['year']}-{meta['month']:02d}-{meta['day']:02d} {meta['stage']} {meta['label']} "
        f"@ {meta['clat']:.1f}°N {meta['clon']:.0f}°E   ·   cyan = the block object at 250 hPa\n"
        f"(d) is NOT an error map. This PPVI has the UPPER piece as its only source, so (d) is what "
        f"the LOWER + SURFACE pieces contribute, plus the unbalanced rotational flow no piecewise\n"
        f"inversion can reach.  RMS$|\\mathbf{{V}}'|$ = {rms(SPD['ref']):.1f} (a) / "
        f"{rms(SPD['block']):.1f} (b) / {rms(SPD['eddy']):.1f} (c) / {rms(SPD['clos']):.1f} (d) "
        f"m s$^{{-1}}$; corr$(\\mathbf{{V}}_{{tot}},\\mathbf{{V}}'_{{rot}})$ = {ctr:.3f}, "
        f"amplitude ratio {rtr:.2f}.\n"
        f"$\\mathbf{{V}}_{{tot}}$ is the INVERSION's own total (upper-piece PV, unmasked) — it is "
        f"NOT $\\mathbf{{V}}'_{{rot}}$. So $\\mathbf{{V}}_p+\\mathbf{{V}}_e-\\mathbf{{V}}_"
        f"{{tot}}$ = {ident:.0e} m s$^{{-1}}$ is an identity about the SPLIT, not a statement "
        f"about realism.",
        fontsize=9.0)
    fig.subplots_adjust(top=0.855, left=0.05, right=0.90, bottom=0.065, hspace=0.12, wspace=0.075)
    _evt.side_colorbar(fig, cf, axes,
                       "$q'$ that induced each wind [PVU]   ((a) and (d) show the total)",
                       shrink=0.80)
    out = HERE / "03_winds_250.png"
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out.name}")
    print(f"  event   : m{meta['member']} {meta['year']}-{meta['month']:02d}-{meta['day']:02d} "
          f"{meta['stage']} {meta['label']}  object {info['levels']} hPa")
    for k, nm in (("ref", "V_rot (external)"), ("block", "V_block        "),
                  ("eddy", "V_eddy         "), ("clos", "closure vs ref ")):
        print(f"  {nm}: RMS = {rms(SPD[k]):6.3f}   max = {np.nanmax(SPD[k][:, ix]):6.3f} m/s")
    print(f"  V_tot  (the INVERSION's own total, NOT V_rot) : RMS = "
          f"{rtr*rms(SPD['ref']):6.3f} m/s   ratio to V_rot = {rtr:.3f}   corr = {ctr:.4f}")
    print(f"  RMS(closure)/RMS(ref) = {fr:.3f}   <- the lower+surface+unbalanced share, not an error")
    print(f"  identity |V_p+V_e-V_tot|max = {ident:.3e} m/s   <- about the SPLIT only")


if __name__ == "__main__":
    main()
