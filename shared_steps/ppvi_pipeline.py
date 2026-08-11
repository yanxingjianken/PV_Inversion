#!/usr/bin/env python3
"""One PPVI run over one event, five steps, one PNG each.

Shared by wu_era / wu_cesm / wu_jra: the datasets differ only in `loaders.py`,
so the pipeline itself is written once.

    01_event_state     the event and mean state, and the inversion box
    02_wu_pv_anomaly   Wu PV anomaly from pass A/B, maps and a cross-section
    03_scale_split     q'_k4, the object mask, q_p and q_e
    04_ppvi_pieces     psi' by piece: surface / lower / upper_p / upper_e
    05_closure         sum of pieces vs the Helmholtz rotational-wind anomaly

Pieces (9-level Wu grid, K is 1-based):

    surface   K=1        1000 hPa bottom boundary theta
    lower     K=2,3,4    850, 700, 500 hPa
    upper_p   K=5..9     400-200 interior PV + 100 hPa top theta, planetary part
    upper_e   K=5..9     the same levels, everything else

upper_p and upper_e are separated by `qp_anoms` / `th_anoms`, not by level, so
they share a level list and differ only in the PV/theta anomaly handed to pass D.

Usage:  python ppvi_pipeline.py --dataset cesm6h --date 1995-01-18T06 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "wu_cesm", "scale_split_sanity"))

from loaders import BAND_S, BAND_N, LOADERS, WU9_HPA          # noqa: E402
from scale_split import (KMAX, KMIN, THRESH, seed_from_centroid,  # noqa: E402
                         split_planetary_eddy, wu_state_band, zonal_filter)
from pvtend.ppvi.solver import invert_piecewise                # noqa: E402
from pvtend.ppvi.winds import psi_to_winds                     # noqa: E402
from pvtend.sh_ops import helmholtz_sh                         # noqa: E402

K_SURFACE, K_LOWER = 0, [1, 2, 3]
K_UPPER_INT, K_TOP = [4, 5, 6, 7], 8
PIECES = {"surface": [1], "lower": [2, 3, 4],
          "upper_p": [5, 6, 7, 8, 9], "upper_e": [5, 6, 7, 8, 9]}
INV_LON_HALF = 90.0        # inversion box: large, so the lateral BC stays away
PATCH_LAT_HALF = 30.0      # event patch: the object search box and the npz window
PATCH_LON_HALF = 60.0
CLIM_HALF_DAYS, CLIM_STRIDE_H = 15, 24     # daily sampling over +-15 d -> 31 samples
KSHOW = 6                                   # 250 hPa


def _status(out, **kw):
    p = os.path.join(out, "STATUS.json")
    d = json.load(open(p)) if os.path.exists(p) else {}
    d.update(kw); d["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(d, open(p, "w"), indent=1)


def _mean_state(loader, when, member, out):
    acc, n = None, 0
    for off in range(-CLIM_HALF_DAYS, CLIM_HALF_DAYS + 1):
        t = when + pd.Timedelta(days=off)
        try:
            s = loader(t, member)
        except Exception:
            continue
        cur = {k: s[k] for k in ("H", "T", "U", "V", "U_glob", "V_glob")}
        acc = cur if acc is None else {k: acc[k] + cur[k] for k in acc}
        n += 1
        if n % 10 == 0:
            _status(out, mean_samples=n)
    if not n:
        raise SystemExit("mean state: no samples")
    return {k: v / n for k, v in acc.items()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    ap.add_argument("--date", required=True)
    ap.add_argument("--member", type=int, default=None)
    ap.add_argument("--clat", type=float, required=True)
    ap.add_argument("--clon", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    when = pd.Timestamp(a.date)
    loader = LOADERS[a.dataset]
    member = a.member if a.member is not None else (91 if a.dataset == "cesm6h" else 1)
    _status(a.out, dataset=a.dataset, date=a.date, member=member, step="01 loading")

    ev = loader(when, member)
    lon, blat = ev["lon"], ev["band_lats"]
    mn, nclim = _mean_state(loader, when, member, a.out)
    _status(a.out, step="01 done", mean_samples=nclim, elapsed_s=round(time.time() - t0))

    clon360 = a.clon % 360.0
    jc = int(np.argmin(np.abs(blat - a.clat)))
    ic = int(np.argmin(np.abs(((lon - clon360 + 180) % 360) - 180)))
    # TWO different boxes, and they must not be confused:
    #   box_*    the INVERSION box, +-90 deg lon x the whole 10.5-85.5 N band.
    #            Large on purpose: psi on its edge is prescribed by the line
    #            integration, so the boundary has to sit far from the object.
    #   patch_*  the EVENT patch, +-60 lon x +-30 lat.  This is where the object
    #            search happens and what the npz stores.  Searching the connected
    #            component in the inversion box instead lets the object run 90 deg
    #            away and across the full 75 deg band -- which is what pushed the
    #            mask to 15-24 % of the domain.
    pad = int(round(INV_LON_HALF / ev["dlon"]))
    box_lat = np.arange(len(blat))
    box_lon = (np.arange(2 * pad + 1) + (ic - pad)) % len(lon)

    plat = int(round(PATCH_LAT_HALF / ev["dlat"]))
    plon = int(round(PATCH_LON_HALF / ev["dlon"]))
    patch_lat = np.arange(max(jc - plat, 0), min(jc + plat + 1, len(blat)))
    patch_lon = (np.arange(2 * plon + 1) + (ic - plon)) % len(lon)

    # ── 01 ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.8), constrained_layout=True)
    for x, (f, t) in zip(ax, ((ev["H"][KSHOW], "event Z 250 hPa [m]"),
                              (mn["H"][KSHOW], f"mean Z ({nclim} samples)"),
                              (ev["H"][KSHOW] - mn["H"][KSHOW], "Z anomaly"))):
        m = np.nanpercentile(np.abs(f - np.nanmean(f)), 99)
        c = np.nanmean(f) if "anomaly" not in t else 0.0
        im = x.pcolormesh(lon, blat, f, cmap="RdBu_r", vmin=c - m, vmax=c + m, shading="auto")
        x.plot(clon360, a.clat, "k*", ms=14)
        x.axvline(lon[box_lon[0]], color="k", lw=.8); x.axvline(lon[box_lon[-1]], color="k", lw=.8)
        x.set_title(t, fontsize=10); fig.colorbar(im, ax=x)
    fig.suptitle(f"01 — {a.dataset} {a.date} m{member}   box ±{INV_LON_HALF:.0f}° lon", fontsize=12)
    fig.savefig(f"{a.out}/01_event_state.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    _status(a.out, step="02 pass A/B")

    # ── 02 ──────────────────────────────────────────────────────────────────
    q_e, _, tht_e = wu_state_band(ev["H"], ev["T"], ev["U"], ev["V"],
                                  blat[0], blat[-1], ev["dlat"], ev["dlon"], ev["nx"])
    q_m, _, tht_m = wu_state_band(mn["H"], mn["T"], mn["U"], mn["V"],
                                  blat[0], blat[-1], ev["dlat"], ev["dlon"], ev["nx"])
    q = q_e - q_m
    th_top = np.asarray(tht_e) - np.asarray(tht_m)

    fig, ax = plt.subplots(1, 3, figsize=(16, 3.8), constrained_layout=True)
    m = np.nanpercentile(np.abs(q[K_UPPER_INT]), 99)
    for x, (f, t) in zip(ax, ((q[3], "Wu PV anom 500 hPa"), (q[KSHOW], "250 hPa"),
                              (q[7], "200 hPa"))):
        im = x.pcolormesh(lon, blat, f, cmap="RdBu_r", vmin=-m, vmax=m, shading="auto")
        x.plot(clon360, a.clat, "k*", ms=12); x.set_title(t, fontsize=10)
        fig.colorbar(im, ax=x)
    fig.suptitle(f"02 — Wu PV anomaly (event − mean), {a.dataset}", fontsize=12)
    fig.savefig(f"{a.out}/02_wu_pv_anomaly.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    _status(a.out, step="03 scale split")

    # ── 03 ──────────────────────────────────────────────────────────────────
    qk_int = zonal_filter(q[K_UPPER_INT], KMIN, KMAX)
    ks = K_UPPER_INT.index(KSHOW)
    js, is_ = seed_from_centroid(qk_int[ks], blat, lon, a.clat, clon360,
                                 radius=min(PATCH_LAT_HALF, PATCH_LON_HALF) / 2)
    r = split_planetary_eddy(q, th_top, K_UPPER_INT, K_TOP,
                             patch_lat, patch_lon, KSHOW, js, is_)
    mk = r["mask"]

    fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for x, (f, t, cm) in zip(ax.ravel(),
                             ((qk_int[ks], f"q'_k4  (k={KMIN}..{KMAX})", "RdBu_r"),
                              (mk[KSHOW] * 1.0, "object mask M", "Greys"),
                              (r["q_p"][KSHOW], "q_p  planetary", "RdBu_r"),
                              (r["q_e"][KSHOW], "q_e  eddy", "RdBu_r"))):
        kw = dict(vmin=0, vmax=1) if cm == "Greys" else dict(vmin=-m, vmax=m)
        im = x.pcolormesh(lon, blat, f, cmap=cm, shading="auto", **kw)
        x.contour(lon, blat, mk[KSHOW], levels=[.5], colors="k", linewidths=.9)
        x.plot(lon[is_], blat[js], "g*", ms=14); x.plot(clon360, a.clat, "k*", ms=10)
        x.set_title(t, fontsize=10); fig.colorbar(im, ax=x)
    inpatch = mk[np.ix_(K_UPPER_INT, patch_lat, patch_lon)]
    fig.suptitle(f"03 — scale split at 250 hPa   mask = {100*inpatch.mean():.1f}% of the "
                 f"±{PATCH_LON_HALF:.0f}°×±{PATCH_LAT_HALF:.0f}° event patch   "
                 f"(green ★ seed, black ★ tracked centroid)", fontsize=12)
    fig.savefig(f"{a.out}/03_scale_split.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    _status(a.out, step="04 inversion",
            mask_frac=float(mk[np.ix_(K_UPPER_INT, patch_lat, patch_lon)].mean()),
            patch=[len(patch_lat), len(patch_lon)], box=[len(box_lat), len(box_lon)])

    # ── 04 ──────────────────────────────────────────────────────────────────
    crop = lambda x: np.ascontiguousarray(x[:, box_lat, :][:, :, box_lon])
    crop2 = lambda x: np.ascontiguousarray(x[box_lat, :][:, box_lon])
    zhdr = np.array([blat[-1], 0.0, blat[0], (len(box_lon) - 1) * ev["dlon"],
                     ev["dlat"], ev["dlon"], len(box_lon), len(box_lat)], dtype=np.float64)
    zero3 = np.zeros((len(WU9_HPA), len(box_lat), len(box_lon)))
    zero2 = np.zeros((len(box_lat), len(box_lon), 2))
    th2 = lambda t: np.stack([np.zeros_like(t), t], axis=2)

    res = invert_piecewise(
        crop(mn["H"]), crop(mn["T"]), crop(mn["U"]), crop(mn["V"]),
        crop(ev["H"]), crop(ev["T"]), crop(ev["U"]), crop(ev["V"]), zhdr,
        pieces=PIECES,
        qp_anoms={"upper_p": crop(r["q_p"]), "upper_e": crop(r["q_e"])},
        th_anoms={"upper_p": th2(crop2(r["th_p"])), "upper_e": th2(crop2(r["th_e"]))},
    )
    psi_p = res["psi_pieces"]
    psi_tot = np.asarray(res["psi_event"]) - np.asarray(res["psi_mean"])

    fig, ax = plt.subplots(1, 5, figsize=(22, 3.6), constrained_layout=True)
    mm = np.nanpercentile(np.abs(psi_tot[KSHOW]), 99)
    for x, (f, t) in zip(ax, [(psi_tot[KSHOW], "total ψ′ (event−mean)")] +
                         [(np.asarray(psi_p[n])[KSHOW], n) for n in PIECES]):
        im = x.pcolormesh(np.arange(len(box_lon)), blat, f, cmap="RdBu_r",
                          vmin=-mm, vmax=mm, shading="auto")
        x.set_title(f"{t}   RMS {np.sqrt(np.nanmean(f**2)):.3g}", fontsize=9)
        fig.colorbar(im, ax=x)
    fig.suptitle(f"04 — balanced ψ′ at 250 hPa by piece [m²/s], {a.dataset}", fontsize=12)
    fig.savefig(f"{a.out}/04_ppvi_pieces.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    _status(a.out, step="05 closure")

    # ── 05 ──────────────────────────────────────────────────────────────────
    # Helmholtz on the GLOBAL sphere, then cut to the band -- see loaders._pack
    band = ev["band"]
    ur_e, vr_e = _helm(ev["U_glob"], ev["V_glob"], ev["lat"], lon)
    ur_m, vr_m = _helm(mn["U_glob"], mn["V_glob"], ev["lat"], lon)
    u_obs = crop(ur_e[:, band, :]); v_obs = crop(vr_e[:, band, :])
    u_obs = u_obs - crop(ur_m[:, band, :]); v_obs = v_obs - crop(vr_m[:, band, :])
    us, vs = {}, {}
    for n in PIECES:
        us[n], vs[n] = psi_to_winds(np.asarray(psi_p[n]), blat, ev["dlat"], ev["dlon"])
    u_sum = sum(us[n] for n in PIECES); v_sum = sum(vs[n] for n in PIECES)

    ok = np.isfinite(u_obs[KSHOW]) & np.isfinite(u_sum[KSHOW])
    r2 = float(np.corrcoef(u_obs[KSHOW][ok], u_sum[KSHOW][ok])[0, 1] ** 2)
    fig, ax = plt.subplots(1, 4, figsize=(19, 3.8), constrained_layout=True)
    mv = np.nanpercentile(np.abs(u_obs[KSHOW]), 99)
    for x, (f, t) in zip(ax, ((u_obs[KSHOW], "observed u'_rot (Helmholtz)"),
                              (u_sum[KSHOW], "Σ pieces"),
                              (u_sum[KSHOW] - u_obs[KSHOW], "Σ − observed"),
                              (us["upper_p"][KSHOW], "upper_p only"))):
        im = x.pcolormesh(np.arange(len(box_lon)), blat, f, cmap="RdBu_r",
                          vmin=-mv, vmax=mv, shading="auto")
        x.set_title(f"{t}   RMS {np.sqrt(np.nanmean(f**2)):.2f} m/s", fontsize=9)
        fig.colorbar(im, ax=x)
    fig.suptitle(f"05 — closure at 250 hPa, R²(Σ, obs) = {r2:.3f}   {a.dataset}", fontsize=12)
    fig.savefig(f"{a.out}/05_closure.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    summ = {n: float(np.sqrt(np.nanmean(us[n][KSHOW] ** 2))) for n in PIECES}
    _status(a.out, step="done", elapsed_s=round(time.time() - t0), r2_closure=r2,
            rms_u_obs=float(np.sqrt(np.nanmean(u_obs[KSHOW] ** 2))),
            rms_u_sum=float(np.sqrt(np.nanmean(u_sum[KSHOW] ** 2))),
            rms_u_by_piece=summ, seed=[int(js), int(is_)])
    print(json.dumps(json.load(open(f"{a.out}/STATUS.json")), indent=1), flush=True)


def _helm(U, V, lat, lon):
    """Rotational part of the wind, level by level (`helmholtz_sh` returns a dict)."""
    ur = np.empty_like(U); vr = np.empty_like(V)
    for k in range(U.shape[0]):
        d = helmholtz_sh(U[k], V[k], lat, lon)
        ur[k], vr[k] = d["u_rot"], d["v_rot"]
    return ur, vr


if __name__ == "__main__":
    try:
        main()
    except Exception:
        out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "."
        os.makedirs(out, exist_ok=True)
        _status(out, step="FAILED", error=traceback.format_exc()[-2000:])
        raise
