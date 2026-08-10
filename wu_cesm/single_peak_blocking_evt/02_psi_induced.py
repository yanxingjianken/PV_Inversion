#!/usr/bin/env python3
"""02 — the streamfunction each PV piece induces, in y-p.

Left  : psi induced by `pv_anom_p`  (the block object, k<=4, 400-200 hPa)
Right : psi induced by `pv_anom_e`  (everything else)

WHY y-p AND NOT x-y
-------------------
PPVI is an ELLIPTIC problem. The source is confined to 400-200 hPa, but the response is not: a PV
anomaly aloft induces flow at levels where it has no PV at all, over a penetration depth ~ f L / N.
That "action at a distance" is the whole point of a piecewise inversion and it is invisible in an
x-y map. So each panel shades the RESPONSE (psi) and contours the SOURCE (the PV anomaly that was
inverted) on top of it — the vertical offset between the two is the physics.

BOTH PANELS SHARE ONE COLOUR SCALE. Per-panel normalisation would make the block- and eddy-induced
circulations look comparable no matter what their actual amplitudes were; the shared scale is what
makes the comparison mean something.

The panels stop at 200 hPa — 100 hPa is not an object level and its archive PV is ~47% missing.

Reads `_cache.npz` written by 01 (rerun 01 --rebuild if the spec changes).

Run:  micromamba run -n blocking python 02_psi_induced.py [--lon-half 0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _evt
from pvtend.ppvi.winds import psi_to_winds

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_cache.npz"
GRID = 1.5
PSI_UNIT = 1e6                      # m^2 s^-1 -> 10^6 m^2 s^-1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lon-half", type=float, default=0.0,
                    help="average over +-this many deg of relative longitude (0 = single column)")
    args = ap.parse_args()

    if not CACHE.exists():
        raise SystemExit("no _cache.npz — run 01_pv_anom_sections.py first")
    z = np.load(CACHE, allow_pickle=True)
    meta, info = eval(str(z["meta"])), eval(str(z["info"]))
    icen = int(z["icen"])
    band, clat = z["band"], float(meta["clat"])
    rel_lat = band - clat
    y = rel_lat[::-1]                                   # ascending latitude for plotting
    WU = np.array(_evt.WU, float)
    NP = len(WU) - 1                                    # 1000..200
    WUP = WU[:NP]

    h = max(1, int(round(args.lon_half / GRID)))
    csl = slice(icen - h + 1, icen + h) if args.lon_half > 0 else slice(icen, icen + 1)
    sec = lambda a: np.nanmean(a[..., csl], axis=-1)[..., ::-1][:NP]

    psi = {"p": sec(z["psi_p"]) / PSI_UNIT, "e": sec(z["psi_e"]) / PSI_UNIT}
    src = {"p": sec(z["pv_anom_p"]), "e": sec(z["pv_anom_e"])}

    # one scale for both panels — see the docstring
    vmax = float(np.nanmax([np.nanpercentile(np.abs(v), 99.5) for v in psi.values()]))
    lev = np.linspace(-vmax, vmax, 21)
    # source contours: also shared, so "the eddy source is stronger" is readable as such
    smax = float(np.nanmax([np.nanpercentile(np.abs(v), 99) for v in src.values()]))
    sl = smax * np.array([0.2, 0.4, 0.6, 0.8])

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.4), sharey=True)
    titles = {"p": r"$\mathbf{(a)}$  $\psi$ induced by $q'_p$  (block object, $k\leq4$)",
              "e": r"$\mathbf{(b)}$  $\psi$ induced by $q'_e$  (everything else)"}

    for ax, key in zip(axes, ("p", "e")):
        cf = ax.contourf(y, WUP, psi[key], levels=lev, cmap="PuOr_r", extend="both")
        ax.contour(y, WUP, psi[key], levels=lev[::4], colors="0.35", linewidths=0.5)
        # the SOURCE, on top of the RESPONSE
        with np.errstate(invalid="ignore"):
            ax.contour(y, WUP, src[key], levels=-sl[::-1], colors="#0b5394", linewidths=1.3)
            ax.contour(y, WUP, src[key], levels=sl, colors="#a61c00", linewidths=1.3,
                       linestyles=":")
        # mark the levels that ACTUALLY carry source, not the nominal piece boundary — the object
        # spans only part of the upper piece and that is exactly what makes the penetration visible
        ax.axhspan(info["levels"][-1], info["levels"][0], color="k", alpha=0.12, zorder=0)
        for pl in (info["levels"][0], info["levels"][-1]):
            ax.axhline(pl, color="k", lw=0.9, ls="--", alpha=0.55)
        ax.axvline(0, color="k", lw=0.7, alpha=0.4)
        ax.set_yscale("log"); ax.set_ylim(1000, 200); ax.set_yticks(WUP)
        ax.set_yticklabels([f"{int(p)}" for p in WUP], fontsize=8); ax.minorticks_off()
        ax.set_xlabel("relative latitude [deg]")
        ax.set_title(titles[key], fontsize=11)
    axes[0].set_ylabel("pressure [hPa]")
    axes[0].text(0.03, 0.045,
                 f"grey band = the ONLY levels with source ({info['levels'][0]}–"
                 f"{info['levels'][-1]} hPa)", transform=axes[0].transAxes, fontsize=8,
                 color="0.2")

    # How far below the source does the response reach? Measure it on the WIND, not on psi.
    # psi carries a large-scale component that never enters the budget; the budget sees V = k x grad psi.
    def wind_profile(key):
        u, v = psi_to_winds(z[f"psi_{key}"], z["band"], GRID, GRID)
        return np.array([np.sqrt(np.nanmean(np.hypot(u[i], v[i]) ** 2)) for i in range(len(_evt.WU))])
    WP = {k: wind_profile(k) for k in ("p", "e")}
    fb = {k: float(WP[k][0] / WP[k].max()) for k in WP}         # 1000 hPa vs the column maximum

    fig.suptitle(
        f"02 · $\\psi$ induced by each PV piece — m{meta['member']} "
        f"{meta['year']}-{meta['month']:02d}-{meta['day']:02d} {meta['stage']} {meta['label']} "
        f"@ {meta['clat']:.1f}°N {meta['clon']:.0f}°E\n"
        f"Shading = the RESPONSE $\\psi$; contours = the SOURCE $q'$ that was inverted "
        f"(blue solid −, red dotted +), on scales shared between the panels.\n"
        f"The source sits at {info['levels'][0]}–{info['levels'][-1]} hPa, yet RMS$|\\mathbf{{V}}|$ "
        f"at 1000 hPa — where there is no source at all — is still\n{100*fb['p']:.0f}% (block) / "
        f"{100*fb['e']:.0f}% (eddy) of its column maximum: the block-induced circulation is "
        f"essentially equivalent-barotropic, the classic observed structure of blocking.",
        fontsize=9.5)
    fig.subplots_adjust(top=0.805, left=0.055, right=0.90, bottom=0.115, wspace=0.08)
    _evt.side_colorbar(fig, cf, axes,
                       r"induced $\psi'$  [$10^6$ m$^2$ s$^{-1}$]  (shared scale)")
    out = HERE / "02_psi_induced.png"
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out.name}")
    print(f"  event      : m{meta['member']} {meta['year']}-{meta['month']:02d}-"
          f"{meta['day']:02d} {meta['stage']} {meta['label']}  object {info['levels']} hPa")
    print(f"  {'hPa':>7s} {'RMS|V_p|':>10s} {'RMS|V_e|':>10s}   (m/s; source = 300-200)")
    for i, pr in enumerate(_evt.WU):
        tag = "  <- source" if pr in info["levels"] else ""
        print(f"  {pr:7d} {WP['p'][i]:10.3f} {WP['e'][i]:10.3f}{tag}")
    print(f"  1000 hPa / column max:  block {fb['p']:.2f}   eddy {fb['e']:.2f}")
    for k, nm in (("p", "block"), ("e", "eddy ")):
        print(f"  psi_{k} ({nm}): p99|psi| = "
              f"{np.nanpercentile(np.abs(z[f'psi_{k}']),99)/PSI_UNIT:7.3f} e6 m2/s")
    r = np.nanpercentile(np.abs(z["psi_p"]), 99) / np.nanpercentile(np.abs(z["psi_e"]), 99)
    print(f"  amplitude ratio block/eddy = {r:.2f}")


if __name__ == "__main__":
    main()
