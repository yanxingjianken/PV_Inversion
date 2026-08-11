#!/usr/bin/env python3
"""Anatomy of the planetary/eddy split — does the second filter blur the edge?

Shows every intermediate of

    q'  ->  q'_k4 = F(q')  ->  M  ->  q'_k4 * M  ->  q_p = F(q'_k4 * M)

side by side, plus what `F` does to the object edge and how far the field leaks
outside the mask.  The two candidate definitions are

    (b)  q_p = q'_k4 * M        confined to the object, NOT k<=4 (~64 % of power)
    (a)  q_p = F(q'_k4 * M)     exactly k<=4, but blurred and leaking outside

so this figure is what decides between them rather than an argument.

Run:  python 00_split_anatomy.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _evt import K_TOP, K_UPPER_INT, WU9_HPA, event_label, load_event
import pandas as pd

from scale_split import (KMAX, KMIN, THRESH, seed_from_centroid,
                         split_planetary_eddy, zonal_filter)

OUT = os.path.dirname(os.path.abspath(__file__))
KSHOW = 6            # 0-based index into WU9 -> 250 hPa


def main():
    ev = load_event()
    lon, blat = ev["lon"], ev["band_lats"]
    q = ev["q_anom"]
    dom = np.array(K_UPPER_INT + [K_TOP])

    # the tracked anticyclone centroid -> the seed that picks THE object
    df = pd.read_csv("/glade/u/home/kenyan/work/01_blocking_cesm2/"
                     "LENS2_z500_anticyclone_blocking.csv", parse_dates=["timestamp"])
    pk = (df[(df.type == "peak") & (df.ens_member == 91)]
          .sort_values("area", ascending=False).iloc[0])
    clat, clon = float(pk.lat), float(pk.lon180) % 360.0
    jc = int(np.argmin(np.abs(blat - clat)))
    ic = int(np.argmin(np.abs(((lon - clon + 180) % 360) - 180)))

    # event box: the band in latitude, +-60 deg in longitude around the centroid
    pad = int(round(60.0 / (lon[1] - lon[0])))
    box_lat = np.arange(len(blat))
    box_lon = (np.arange(2 * pad + 1) + (ic - pad)) % len(lon)

    q_k4_int = zonal_filter(q[K_UPPER_INT], KMIN, KMAX)
    kshow_int = K_UPPER_INT.index(KSHOW)
    js, is_ = seed_from_centroid(q_k4_int[kshow_int], blat, lon, clat, clon)
    print(f"  centroid {clat:.1f}N {clon:.1f}E (j={jc},i={ic}) -> seed "
          f"{blat[js]:.1f}N {lon[is_]:.1f}E (j={js},i={is_}), "
          f"q_k4 there = {q_k4_int[kshow_int][js,is_]:.1f}", flush=True)

    r = split_planetary_eddy(q, ev["th_top_anom"], K_UPPER_INT, K_TOP,
                             box_lat, box_lon, KSHOW, js, is_)
    mask_full = r["mask"]
    q_k4 = zonal_filter(q[dom], KMIN, KMAX)
    mask = mask_full[dom]
    q_masked = q_k4 * mask
    q_p = zonal_filter(q_masked, KMIN, KMAX)
    q_e_full = q[dom] - q_p

    ks = list(dom).index(KSHOW)
    panels = [
        (q[KSHOW],        "q'  (raw anomaly)"),
        (q_k4[ks],        f"q'_k4 = F(q'),  k={KMIN}..{KMAX}"),
        (mask[ks] * 1.0,  f"M  (3-D connected, q'_k4 < -{THRESH})"),
        (q_masked[ks],    "(b)  q'_k4 · M"),
        (q_p[ks],         "(a)  q_p = F(q'_k4 · M)"),
        (q_p[ks] - q_masked[ks], "(a) − (b)   = what F changed"),
    ]

    m = float(np.nanpercentile(np.abs(q_k4), 99))
    fig, axes = plt.subplots(3, 2, figsize=(15, 10.5), constrained_layout=True)
    for ax, (fld, ttl) in zip(axes.ravel(), panels):
        if "M  (" in ttl:
            im = ax.pcolormesh(lon, blat, fld, cmap="Greys", vmin=0, vmax=1,
                               shading="auto")
        else:
            im = ax.pcolormesh(lon, blat, fld, cmap="RdBu_r", vmin=-m, vmax=m,
                               shading="auto")
        ax.contour(lon, blat, mask[ks], levels=[0.5], colors="k", linewidths=1.0)
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel("lon [°E]"); ax.set_ylabel("lat [°N]")
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"00 — split anatomy at {WU9_HPA[KSHOW]:.0f} hPa   ({event_label(ev['cache'])})\n"
                 "black contour = the object mask M in every panel", fontsize=13)
    fig.savefig(f"{OUT}/00_split_anatomy.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── how much leaks outside the mask, and how blurred is the edge ────────
    inside = mask[ks]
    outside = ~inside
    p_in = float(np.sum(q_p[ks][inside] ** 2))
    p_out = float(np.sum(q_p[ks][outside] ** 2))
    b_in = float(np.sum(q_masked[ks][inside] ** 2))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
    j = int(np.argmax(np.abs(q_p[ks]).max(axis=1)))
    ax[0].plot(lon, q[KSHOW][j], color="0.6", lw=1, label="q'  raw")
    ax[0].plot(lon, q_k4[ks][j], "k--", lw=1.2, label="q'_k4")
    ax[0].plot(lon, q_masked[ks][j], "C1", lw=1.6, label="(b) q'_k4·M")
    ax[0].plot(lon, q_p[ks][j], "C0", lw=1.8, label="(a) F(q'_k4·M)")
    ax[0].fill_between(lon, -m, m, where=inside[j], color="k", alpha=0.08,
                       label="inside M")
    ax[0].set_xlabel("lon [°E]"); ax[0].set_ylabel("Wu PV anomaly")
    ax[0].set_title(f"cut at {blat[j]:.0f}°N — the edge", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].set_ylim(-m, m)

    from scale_split import wavenumber_spectrum
    for lbl, fld, c in (("q'", q[dom], "0.6"), ("q'_k4", q_k4, "k"),
                        ("(b) q'_k4·M", q_masked, "C1"), ("(a) F(q'_k4·M)", q_p, "C0")):
        s = wavenumber_spectrum(fld)
        ax[1].semilogy(np.arange(len(s))[:25], np.maximum(s[:25], 1e-12), "o-",
                       color=c, ms=3.5, lw=1.3, label=lbl)
    ax[1].axvline(KMAX + 0.5, color="r", ls=":", lw=1.2)
    ax[1].set_xlabel("zonal wavenumber"); ax[1].set_ylabel("fractional power")
    ax[1].set_title(f"spectra (red = k>{KMAX} boundary)", fontsize=10)
    ax[1].legend(fontsize=8); ax[1].set_ylim(1e-8, 1)

    labels = ["power inside M", "power outside M\n(leakage)"]
    ax[2].bar([0, 1], [100 * p_in / (p_in + p_out), 100 * p_out / (p_in + p_out)],
              color=["C0", "C3"])
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(labels, fontsize=9)
    ax[2].set_ylabel("% of (a) power"); ax[2].set_ylim(0, 100)
    ax[2].set_title(f"(a) amplitude inside M is "
                    f"{100*np.sqrt(p_in/max(b_in,1e-30)):.0f} % of (b)'s", fontsize=10)
    fig.suptitle("00b — what the second filter costs: edge, spectrum, leakage",
                 fontsize=12)
    fig.savefig(f"{OUT}/00b_split_cost.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    s_b = wavenumber_spectrum(q_masked)
    s_a = wavenumber_spectrum(q_p)
    print(f"event: {event_label(ev['cache'])}   level {WU9_HPA[KSHOW]:.0f} hPa")
    print(f"  mask: {100*mask.mean():.2f}% of the upper domain, "
          f"reaches top boundary: {bool(mask[-1].any())} "
          f"({100*mask[-1].mean():.2f}% of that level)")
    print(f"  (b) q'_k4·M      : k=0 {100*s_b[0]:5.2f}%  k=1-4 {100*s_b[1:5].sum():6.2f}%  "
          f"k>4 {100*s_b[5:].sum():6.2f}%")
    print(f"  (a) F(q'_k4·M)   : k=0 {100*s_a[0]:5.2f}%  k=1-4 {100*s_a[1:5].sum():6.2f}%  "
          f"k>4 {100*s_a[5:].sum():6.2f}%")
    print(f"  (a) power leaking outside M : {100*p_out/(p_in+p_out):.1f}%")
    print(f"  (a) rms inside M / (b) rms inside M : "
          f"{np.sqrt(p_in/max(b_in,1e-30)):.3f}")
    print(f"  additivity |q_p+q_e-q| : "
          f"{np.abs(q_p + q_e_full - q[dom]).max():.3e}")
    print(f"\nwrote {OUT}/00_split_anatomy.png and 00b_split_cost.png")


if __name__ == "__main__":
    main()
