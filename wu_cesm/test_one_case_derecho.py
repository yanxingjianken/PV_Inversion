#!/usr/bin/env python3
"""Single-event Wu PPVI feasibility test on Derecho — CESM2-LENS2 smbb, NL=20.

Derecho port of `test_one_case.py`, which reads dolma paths that do not exist
here (a precomputed 9-level PV netCDF and a precomputed daily climatology).
Both inputs are instead built from the native `hour_6 cam.h5` archive through the
same code that feeds the production sweep, so this test exercises the real path.

Level set is the **20-level** Wu grid (σ = 1.00…0.05, i.e. 1000…50 hPa), matching
the freshly rebuilt NL=20 f2py extension.

Pieces are the 3-way grouping, not the per-level default: with 20 per-level pieces
this would be 20 pass-D inversions, and the question here is only whether the
machinery runs and closes.

    surface  K=1        1000 hPa boundary θ  (Bretherton warm core)
    lower    K=2..11    950–500 hPa
    upper    K=12..20   450–100 hPa + 50 hPa top θ

**Climatology caveat**: the mean state is a ±15-day window in the SAME year
(120 six-hourly samples), not the 30-year climatology.  That is a deliberate
shortcut so this test does not block on the 300-job archive sweep; it is enough
to exercise the solver but it is NOT the climatology the science should use.

Each step writes a PNG.

Run:  python test_one_case_derecho.py
"""
from __future__ import annotations

import os
import sys
import time

import cftime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
sys.path.insert(0, "/glade/derecho/scratch/kenyan/01_cesm_processing/06_isentropic_clim")
sys.path.insert(0, "/glade/derecho/scratch/kenyan/01_cesm_processing/05_pv_budget_closure")

import lens2_archive as la          # noqa: E402
import state_isobaric as si         # noqa: E402
from pvtend.ppvi.solver import NL, PR, invert_piecewise   # noqa: E402
from pvtend.ppvi.winds import psi_to_winds                # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
BLOCK_CSV = "/glade/u/home/kenyan/work/01_blocking_cesm2/LENS2_z500_anticyclone_blocking.csv"

BAND_S, BAND_N = 10.5, 85.5      # inversion latitude band, as in the dolma test
INV_LON_HALF = 90.0              # pvtend default; leaves 30 deg of buffer outside the +-60 patch
CLIM_HALF_DAYS = 15

PIECES3 = {"surface": [1],
           "lower": list(range(2, 12)),
           "upper": list(range(12, NL + 1))}


def _state(member, k, lat, lon, hyam, hybm, p0):
    F = {v: la.read_slice(la.SMBB, member, v, k) for v in ("U", "V", "T", "Q", "Z3", "OMEGA")}
    ps = la.read_slice(la.SMBB, member, "PS", k)
    return si.state_on_pressure(F, ps, hyam, hybm, p0, lat, lon)


def main():
    t_start = time.time()

    # ── pick the strongest blocking peak of member 91 ───────────────────────
    df = pd.read_csv(BLOCK_CSV, parse_dates=["timestamp"])
    pk = (df[(df.type == "peak") & (df.ens_member == 91)]
          .sort_values("area", ascending=False).iloc[0])
    member = 91
    d = pk.timestamp.to_pydatetime()
    print(f"event: member {member}  {pk.timestamp}  "
          f"{pk.lat:.1f}N {pk.lon180:.1f}E  area {pk.area/1e12:.1f}e12 m2", flush=True)

    lat, lon, hyam, hybm, p0 = la.grid_and_hybrid("smbb", member)
    ti = la.time_index("smbb", member, "U")
    k0 = ti.index_of(cftime.DatetimeNoLeap(d.year, d.month, d.day, d.hour))

    # ── box: full 10.5-85.5N band x +-INV_LON_HALF around the block ─────────
    band = np.nonzero((lat >= BAND_S - 1e-6) & (lat <= BAND_N + 1e-6))[0][::-1]   # N->S
    dlat = float(abs(lat[1] - lat[0]))
    dlon = float(lon[1] - lon[0])
    clon360 = pk.lon180 % 360.0
    ic = int(np.argmin(np.abs(((lon - clon360 + 180) % 360) - 180)))
    pad = int(round(INV_LON_HALF / dlon))
    lon_idx = (np.arange(2 * pad + 1) + (ic - pad)) % len(lon)
    ny, nx = band.size, lon_idx.size
    band_lats = lat[band]
    zhdr = np.array([band_lats[-1], 0.0, band_lats[0], (nx - 1) * dlon,
                     dlat, dlon, nx, ny], dtype=np.float64)
    print(f"box: ny={ny} nx={nx}  lat {band_lats[-1]:.2f}..{band_lats[0]:.2f}N  "
          f"lon half-width {INV_LON_HALF:.0f} deg", flush=True)

    def crop(a):
        return a[:, band, :][:, :, lon_idx]

    # ── event + mean state (cached: the mean costs ~5 min) ──────────────────
    cache = f"{OUT}/_state_cache_m{member}_{d:%Y%m%d%H}.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        ev = {v: z[f"ev_{v}"] for v in ("Z3", "T", "U", "V", "PV")}
        mn = {v: z[f"mn_{v}"] for v in ("Z3", "T", "U", "V")}
        print(f"state loaded from cache ({time.time()-t_start:.0f} s)", flush=True)
        _skip_clim = True
    else:
        _skip_clim = False
    if not _skip_clim:
        ev = _state(member, k0, lat, lon, hyam, hybm, p0)
        print(f"event state built ({time.time()-t_start:.0f} s)", flush=True)

    # ── mean state: +-15 d window, same year, 6-hourly ──────────────────────
    if not _skip_clim:
     ks = [k for k in range(k0 - CLIM_HALF_DAYS * 4, k0 + CLIM_HALF_DAYS * 4 + 1)
          if 0 <= k < len(ti)]
     acc, cnt = None, 0
     for n, k in enumerate(ks):
        st = _state(member, k, lat, lon, hyam, hybm, p0)
        cur = {v: st[v] for v in ("Z3", "T", "U", "V")}
        acc = cur if acc is None else {v: acc[v] + cur[v] for v in acc}
        cnt += 1
        if (n + 1) % 20 == 0:
            print(f"  clim {n+1}/{len(ks)}  ({time.time()-t_start:.0f} s)", flush=True)
     mn = {v: acc[v] / cnt for v in acc}
     np.savez_compressed(cache, **{f"ev_{v}": ev[v] for v in ("Z3","T","U","V","PV")},
                         **{f"mn_{v}": mn[v] for v in mn})
     print(f"mean state: {cnt} samples, cached ({time.time()-t_start:.0f} s)", flush=True)

    H_e, T_e, U_e, V_e = (crop(ev[v]) for v in ("Z3", "T", "U", "V"))
    H_m, T_m, U_m, V_m = (crop(mn[v]) for v in ("Z3", "T", "U", "V"))

    # ── PNG 1: event vs mean, and the box ───────────────────────────────────
    k250 = int(np.argmin(np.abs(si.WU_HPA - 250)))
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6), constrained_layout=True)
    for a, (fld, ttl) in zip(ax, ((ev["PV"][k250], "event PV 250 hPa [PVU]"),
                                  (mn["Z3"][k250] * 0 + ev["Z3"][k250],
                                   "event Z 250 hPa [m]"),
                                  (ev["Z3"][k250] - mn["Z3"][k250],
                                   "Z anomaly 250 hPa [m]"))):
        m = np.nanpercentile(np.abs(fld[96:]), 98)
        im = a.pcolormesh(lon, lat[96:], fld[96:], cmap="RdBu_r", vmin=-m, vmax=m,
                          shading="auto")
        a.plot(pk.lon180 % 360, pk.lat, "k*", ms=14)
        a.axhline(BAND_S, color="k", lw=0.6); a.axhline(BAND_N, color="k", lw=0.6)
        a.set_title(ttl, fontsize=10); fig.colorbar(im, ax=a)
    fig.suptitle(f"01 — event state, member {member} {pk.timestamp}", fontsize=12)
    fig.savefig(f"{OUT}/derecho_01_event_state.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── inversion ───────────────────────────────────────────────────────────
    print("running invert_piecewise (NL=%d, 3 pieces) ..." % NL, flush=True)
    t0 = time.time()
    res = invert_piecewise(H_m, T_m, U_m, V_m, H_e, T_e, U_e, V_e, zhdr)
    # group the per-level pieces afterwards: pass D is LINEAR, so summing
    # levels is exactly equivalent to inverting a grouped piece -- and the
    # refactored solver only supports single-level pieces (a multi-level
    # list silently returned all-NaN).
    psi_p = {g: sum(res["psi_pieces"][str(i)] for i in lv)
             for g, lv in PIECES3.items()}
    print(f"  done in {time.time()-t0:.0f} s   keys: {sorted(res)}", flush=True)

    # the perturbation reference is event - mean; `psi_total` carries an
    # absolute pedestal (RMS 7e8 m2/s, near-uniform) and is not the anomaly
    psi_tot = res["psi_event"] - res["psi_mean"]

    # ── PNG 2: Wu PV anomaly cross-section ──────────────────────────────────
    qa = res["Q_event"] - res["Q_mean"]
    jmid = ny // 2
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    m = np.nanpercentile(np.abs(qa), 99)
    im = ax[0].pcolormesh(np.arange(nx), si.WU_HPA, qa[:, jmid, :], cmap="RdBu_r",
                          vmin=-m, vmax=m, shading="auto")
    ax[0].invert_yaxis(); ax[0].set_ylabel("p [hPa]")
    ax[0].set_title(f"Wu PV anomaly, lat {band_lats[jmid]:.0f}N"); fig.colorbar(im, ax=ax[0])
    im = ax[1].pcolormesh(np.arange(nx), band_lats, qa[k250], cmap="RdBu_r",
                          vmin=-m, vmax=m, shading="auto")
    ax[1].set_title("Wu PV anomaly, 250 hPa"); fig.colorbar(im, ax=ax[1])
    fig.suptitle("02 — Wu PV anomaly that is inverted", fontsize=12)
    fig.savefig(f"{OUT}/derecho_02_pv_anomaly.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── PNG 3: psi per piece at 250 hPa + induced winds ─────────────────────
    fig, ax = plt.subplots(1, 4, figsize=(19, 3.6), constrained_layout=True)
    m = np.nanpercentile(np.abs(psi_tot[k250]), 99)
    for a, (fld, ttl) in zip(ax, [(psi_tot[k250], "total ψ′")] +
                             [(psi_p[n][k250], f"{n} ψ′") for n in PIECES3]):
        im = a.pcolormesh(np.arange(nx), band_lats, fld, cmap="RdBu_r",
                          vmin=-m, vmax=m, shading="auto")
        a.set_title(f"{ttl}  (RMS {np.sqrt(np.nanmean(fld**2)):.3g})", fontsize=10)
        fig.colorbar(im, ax=a)
    fig.suptitle("03 — balanced ψ′ at 250 hPa, total and by piece [m²/s]", fontsize=12)
    fig.savefig(f"{OUT}/derecho_03_psi_pieces.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── PNG 4 + GATE: linearity, Σ pieces vs total ──────────────────────────
    psi_sum = sum(res["psi_pieces"][str(i)] for i in range(1, NL + 1))
    ok = np.isfinite(psi_sum) & np.isfinite(psi_tot)
    r2 = np.corrcoef(psi_sum[ok], psi_tot[ok])[0, 1] ** 2
    nrms = (np.sqrt(np.mean((psi_sum[ok] - psi_tot[ok]) ** 2))
            / np.sqrt(np.mean(psi_tot[ok] ** 2)))

    u_t, v_t = psi_to_winds(psi_tot, band_lats, dlat, dlon)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    ax[0].hexbin(psi_tot[ok], psi_sum[ok], gridsize=60, bins="log", cmap="viridis")
    lim = np.nanpercentile(np.abs(psi_tot[ok]), 99.5)
    ax[0].plot([-lim, lim], [-lim, lim], "r-", lw=1)
    ax[0].set_xlabel("total ψ′"); ax[0].set_ylabel("Σ pieces")
    ax[0].set_title(f"linearity: R² = {r2:.6f}, NRMS = {100*nrms:.3f}%")
    d3 = psi_sum[k250] - psi_tot[k250]
    m = max(np.nanpercentile(np.abs(d3), 99), 1e-12)
    im = ax[1].pcolormesh(np.arange(nx), band_lats, d3, cmap="RdBu_r",
                          vmin=-m, vmax=m, shading="auto")
    ax[1].set_title("Σ pieces − total, 250 hPa"); fig.colorbar(im, ax=ax[1])
    m = np.nanpercentile(np.abs(u_t[k250]), 99)
    im = ax[2].pcolormesh(np.arange(nx), band_lats, u_t[k250], cmap="RdBu_r",
                          vmin=-m, vmax=m, shading="auto")
    ax[2].set_title("induced u′ from total ψ′, 250 hPa [m/s]"); fig.colorbar(im, ax=ax[2])
    fig.suptitle("04 — solver linearity gate", fontsize=12)
    fig.savefig(f"{OUT}/derecho_04_linearity.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nGATE linearity: R² = {r2:.6f}  NRMS = {100*nrms:.4f}%  "
          f"{'PASS' if (r2 > 0.99 and nrms < 0.05) else 'FAIL'}")
    print(f"induced u′ at 250 hPa: RMS {np.sqrt(np.nanmean(u_t[k250]**2)):.3f} m/s, "
          f"max |u′| {np.nanmax(np.abs(u_t[k250])):.2f} m/s")
    for n in PIECES3:
        up, _ = psi_to_winds(psi_p[n], band_lats, dlat, dlon)
        print(f"  piece {n:8s}: induced u′ RMS at 250 hPa = "
              f"{np.sqrt(np.nanmean(up[k250]**2)):.3f} m/s")
    print(f"\ntotal wall {time.time()-t_start:.0f} s; 4 PNGs in {OUT}")


if __name__ == "__main__":
    main()
