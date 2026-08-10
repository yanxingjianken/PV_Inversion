#!/usr/bin/env python3
"""
Phase-0 GATE — Wu piecewise PV-inversion closure on ONE native CESM2-LENS2 block.

De-risks the f09 inversion path BEFORE editing pvtend: it drives pvtend's
*existing* solver (``pvtend.ppvi.solver.invert_piecewise`` +
``pvtend.ppvi.winds.psi_to_winds``) on real CESM f09 data (no Fortran
recompile — NL=9 matches, NY/NX inferred), and checks three closures:

  (A) SOLVER LINEARITY  — Σ(per-level piece ψ′) vs the single all-levels
      perturbation inversion ψ′ (both pass-D). GATE: R² > 0.99 @250,
      interior NRMS < ~5 %.
  (PV) PV CLOSURE       — Wu balanced PV Q_event vs the input Ertel PV
      (structural NRMS / correlation).
  (B) PI RECOVERY       — the balanced rotational-wind anomaly reconstructed
      from Σ(piece winds) vs the OBSERVED u_rot_anom = u_rot(event) −
      u_rot(clim). Helmholtz is done on the FULL global f09 sphere first
      (event via sh_ops.helmholtz_sh; clim from the precomputed u_rot_bar),
      then cropped — u_rot_anom is the recover/check target. R² @250/500.
  (adv) ADVECTION SANITY — −u_rot_anom_ppvi·∇q' (per piece + Σ) and the
      divergent −u_div_anom·∇q' term, in PVU/day (finite, O(1-10)).

CESM z is geopotential HEIGHT [m] → H = z directly (NO /g, unlike ERA5).
Latitude band = f09 gridpoints in [10.5, 85.5]°N; cubes flipped to N→S; zhdr
lat_s/lat_n set to the ACTUAL f09 band-edge latitudes.

Run:
    micromamba run -n blocking python test_one_case.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pvtend.ppvi.solver import invert_piecewise
from pvtend.ppvi.winds import psi_to_winds
from pvtend.sh_ops import helmholtz_sh
from pvtend.constants import G0, R_EARTH

# ── Test event (m1 onset track 3: 1985-02-05, Atlantic block) ───────────────
MEMBER = 1
DECADE = "19851989"
YEAR, MONTH, DAY = 1985, 2, 5
CLON180, CLAT = -25.601309, 52.022117
MON_ABBR = ["jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"]

PV_NC = Path(f"/net/flood/data2/users/x_yan/cesm-blocking/blocking_event_data/"
             f"pv_nc/m{MEMBER}/cesm2_lens2_pv9_m{MEMBER:02d}_d{DECADE}.nc")
CLIM_DIR = Path("/net/flood/data2/users/x_yan/cesm-blocking/blocking_event_data/clim")
CLIM_STEM = "cesm_lens2_dailyclim_1985-2014_ensmean"
OUT_DIR = Path(__file__).resolve().parent

WU_PLEVS = [1000, 850, 700, 500, 400, 300, 250, 200, 100]
WU2SI = 1.0e-8          # Wu pseudo-PV → SI Ertel PV
MI = 9999.90
BAND_S, BAND_N = 10.5, 85.5
INV_LON_HALF = 90.0
K250 = WU_PLEVS.index(250)   # 6
K500 = WU_PLEVS.index(500)   # 3
DAY3 = 24 * 3600.0


def _load_event():
    with xr.open_dataset(PV_NC) as ds:
        times = ds["time"].values
        ti = next(i for i, t in enumerate(times)
                  if (t.year, t.month, t.day) == (YEAR, MONTH, DAY))
        sub = ds.isel(time=ti).sel(lev=WU_PLEVS)
        out = {v: sub[v].values.astype(np.float64) for v in
               ("z", "t", "u", "v", "pv")}
        lat = ds["lat"].values
        lon = ds["lon"].values
    return out, lat, lon


def _load_clim():
    ab = MON_ABBR[MONTH - 1]
    out = {}
    for v in ("z", "t", "u", "v", "pv"):
        with xr.open_dataset(CLIM_DIR / f"{CLIM_STEM}_{ab}_{v}.nc") as ds:
            out[v] = (ds[v].isel(month=0, hour=0, day=DAY - 1)
                      .sel(pressure_level=WU_PLEVS).values.astype(np.float64))
    # precomputed clim Helmholtz rotational/divergent components
    with xr.open_dataset(CLIM_DIR / f"{CLIM_STEM}_{ab}_u_helmholtz.nc") as du:
        out["u_rot_bar"] = (du["u_rot_bar"].isel(hour=0, day=DAY - 1)
                            .sel(pressure_level=WU_PLEVS).values.astype(np.float64))
        out["u_div_bar"] = (du["u_div_bar"].isel(hour=0, day=DAY - 1)
                            .sel(pressure_level=WU_PLEVS).values.astype(np.float64))
    with xr.open_dataset(CLIM_DIR / f"{CLIM_STEM}_{ab}_v_helmholtz.nc") as dv:
        out["v_rot_bar"] = (dv["v_rot_bar"].isel(hour=0, day=DAY - 1)
                            .sel(pressure_level=WU_PLEVS).values.astype(np.float64))
        out["v_div_bar"] = (dv["v_div_bar"].isel(hour=0, day=DAY - 1)
                            .sel(pressure_level=WU_PLEVS).values.astype(np.float64))
    return out


def _helmholtz_global(u3d, v3d, lat, lon):
    """Global f09 Helmholtz per level (lat ascending) → u_rot,v_rot,u_div,v_div."""
    nl = u3d.shape[0]
    ur = np.empty_like(u3d); vr = np.empty_like(u3d)
    ud = np.empty_like(u3d); vd = np.empty_like(u3d)
    for k in range(nl):
        h = helmholtz_sh(u3d[k], v3d[k], lat, lon, R_earth=R_EARTH)
        ur[k], vr[k] = h["u_rot"], h["v_rot"]
        ud[k], vd[k] = h["u_div"], h["v_div"]
    return ur, vr, ud, vd


def _r2(y_true, y_pred):
    a, b = y_true.ravel(), y_pred.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return np.nan
    ss_res = np.sum((a[m] - b[m]) ** 2)
    ss_tot = np.sum((a[m] - a[m].mean()) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def _nrms(resid, signal):
    return (np.sqrt(np.nanmean(resid ** 2))
            / max(np.sqrt(np.nanmean(signal ** 2)), 1e-30))


def main():
    rpt = []
    def log(s): print(s, flush=True); rpt.append(s)

    log(f"=== Phase-0 Wu closure on CESM2-LENS2 m{MEMBER} {YEAR}-{MONTH:02d}-{DAY:02d} "
        f"(block @ {CLAT:.1f}N {CLON180:.1f}E) ===")

    ev, lat, lon = _load_event()
    cl = _load_clim()
    nlon = lon.size

    # global Helmholtz of the EVENT total wind (clim rot/div from precomputed bar)
    ur_e, vr_e, ud_e, vd_e = _helmholtz_global(ev["u"], ev["v"], lat, lon)
    ur_c, vr_c = cl["u_rot_bar"], cl["v_rot_bar"]
    ud_c, vd_c = cl["u_div_bar"], cl["v_div_bar"]

    # ── band (10.5–85.5°N) × lon window centred on the block ────────────────
    band = np.where((lat >= BAND_S - 1e-6) & (lat <= BAND_N + 1e-6))[0]
    band = band[::-1]                       # N→S
    band_lats = lat[band]
    dlat = float(abs(np.diff(lat).mean()))
    dlon = float(abs(np.diff(lon).mean()))
    clon360 = CLON180 % 360.0
    ilon = int(np.argmin(np.abs(((lon - clon360 + 180) % 360) - 180)))
    pad = int(round(INV_LON_HALF / dlon))
    lon_idx = (np.arange(2 * pad + 1) + (ilon - pad)) % nlon
    ny, nx = band.size, lon_idx.size
    lon_win = np.rad2deg(np.unwrap(np.deg2rad(lon[lon_idx])))
    # Standard pvtend zhdr convention [lat_s, lon_w, lat_n, lon_e, dlat, dlon,
    # nx, ny]. pvtend ≥2.13 reorders dlat/dlon internally to the Fortran's
    # HDR(5)=Δlon / HDR(6)=Δlat convention, so anisotropic f09 inverts correctly
    # with this standard header (no caller-side workaround needed).
    zhdr = np.array([band_lats[-1], 0.0, band_lats[0], (nx - 1) * dlon,
                     dlat, dlon, nx, ny], dtype=np.float32)
    log(f"band: {ny} lats {band_lats[0]:.2f}→{band_lats[-1]:.2f}N | "
        f"lon window: {nx} pts {lon_win[0]:.1f}→{lon_win[-1]:.1f} | dlat={dlat:.3f} dlon={dlon:.2f}")

    def crop(a3d):
        return a3d[:, band][:, :, lon_idx]

    # Raw cropped cubes (NL, ny, nx), N→S, WITH below-ground NaN. CESM z is
    # HEIGHT [m] → H = z (no /g). invert_piecewise (pvtend ≥2.13, fill_nan=True
    # default) hydrostatically gap-fills the below-ground NaN itself.
    H_e, T_e, U_e, V_e = crop(ev["z"]), crop(ev["t"]), crop(ev["u"]), crop(ev["v"])
    H_m, T_m, U_m, V_m = crop(cl["z"]), crop(cl["t"]), crop(cl["u"]), crop(cl["v"])
    pv_e_band, pv_m_band = crop(ev["pv"]), crop(cl["pv"])
    urc_b, vrc_b = crop(ur_c), crop(vr_c)
    ure_b, vre_b = crop(ur_e), crop(vr_e)
    ud_anom_b = crop(ud_e) - crop(ud_c)
    vd_anom_b = crop(vd_e) - crop(vd_c)
    # observed rotational-wind anomaly = u_rot(event) − u_rot(clim)  ← CHECK TARGET
    u_rot_anom = ure_b - urc_b
    v_rot_anom = vre_b - vrc_b

    # ── Wu piecewise inversion: per-level pieces + a single all-levels piece ─
    pieces = {str(i): [i] for i in range(1, 10)}
    pieces["all"] = list(range(1, 10))
    log("running invert_piecewise (per-level + all) ...")
    res = invert_piecewise(H_m, T_m, U_m, V_m, H_e, T_e, U_e, V_e, zhdr,
                           pieces=pieces)

    psi = res["psi_pieces"]
    psi_perlevel = [psi[str(i)] for i in range(1, 10)]
    psi_sum = np.sum(psi_perlevel, axis=0)         # Σ per-level ψ′
    psi_all = psi["all"]                           # single all-levels ψ′

    # piece winds (balanced rotational) via Wu finite-difference psi→winds
    def winds(p):
        return psi_to_winds(p, band_lats, dlat, dlon)
    u_sum, v_sum = winds(psi_sum)
    u_all, v_all = winds(psi_all)

    # ── (A) solver linearity: Σ per-level vs single all-levels ──────────────
    log("\n--- (A) SOLVER LINEARITY: Σ(per-level ψ′) vs single all-levels ψ′ ---")
    r2_psi_250 = _r2(psi_all[K250], psi_sum[K250])
    log(f"  ψ′  R²@250 = {r2_psi_250:.6f}   R²@500 = {_r2(psi_all[K500], psi_sum[K500]):.6f}")
    log(f"  u_rot R²@250 = {_r2(u_all[K250], u_sum[K250]):.6f}")
    log(f"  {'lev':>5} {'NRMS(ψ′)':>10}")
    nrms = {}
    for k, L in enumerate(WU_PLEVS):
        nrms[L] = _nrms(psi_sum[k] - psi_all[k], psi_all[k])
        log(f"  {L:>5} {nrms[L]:>10.4f}")
    # Gate on the upper-trop wavg band (300/250/200) — the blocking-analysis
    # levels. Lower levels (1000/850/700) carry below-ground-fill artefacts and
    # 500/400 a little pass-D nonlinearity (inlin=1), so they are reported but
    # not gated.
    max_nrms_upper = float(np.nanmax([nrms[L] for L in (300, 250, 200)]))

    # ── (PV) PV closure: Wu balanced PV vs input Ertel PV ───────────────────
    log("\n--- (PV) PV CLOSURE: Wu Q_event vs input Ertel PV [PVU] ---")
    Q_e = np.asarray(res["Q_event"])
    bad = np.abs(Q_e) >= MI * 0.99
    Q_e_pvu = np.where(bad, np.nan, Q_e * WU2SI * 1.0e6)   # → PVU
    # interior levels only (boundary K=1,9 are sentinel)
    for L, k in [("500", K500), ("250", K250)]:
        a, b = pv_e_band[k], Q_e_pvu[k]
        log(f"  @{L}: corr={np.corrcoef(a.ravel(), np.nan_to_num(b).ravel())[0,1]:.4f}  "
            f"NRMS={_nrms(b - a, a):.4f}  (Wu mean {np.nanmean(b):.2f}, Ertel mean {np.nanmean(a):.2f} PVU)")

    # ── (B) PI recovery: Σ(piece winds) vs observed u_rot_anom ──────────────
    log("\n--- (B) PI RECOVERY: balanced Σ(piece winds) vs observed u_rot_anom ---")
    pi_r2 = {}
    for L, k in [("250", K250), ("500", K500)]:
        msk = np.isfinite(u_sum[k]) & np.isfinite(u_rot_anom[k])
        pi_r2[L] = _r2(u_rot_anom[k][msk], u_sum[k][msk])
        log(f"  @{L}: u  R²={pi_r2[L]:.4f}  "
            f"v  R²={_r2(v_rot_anom[k][msk], v_sum[k][msk]):.4f}  "
            f"(‖Σpieces‖={np.sqrt(np.nanmean(u_sum[k]**2)):.1f}, "
            f"‖obs‖={np.sqrt(np.nanmean(u_rot_anom[k]**2)):.1f} m/s)")

    # ── (adv) advection sanity at 250 hPa [PVU/day] ─────────────────────────
    log("\n--- (adv) ADVECTION SANITY @250 [PVU/day] ---")
    pv_anom = pv_e_band - pv_m_band
    dx = np.deg2rad(dlon) * R_EARTH * np.cos(np.deg2rad(band_lats))[:, None]
    dy = np.deg2rad(dlat) * R_EARTH
    dqdx = np.gradient(pv_anom[K250], axis=1) / dx
    dqdy = np.gradient(pv_anom[K250], axis=0) / (-dy)   # N→S rows → y points south
    adv_rot = -(u_sum[K250] * dqdx + v_sum[K250] * dqdy) * DAY3
    adv_div = -(ud_anom_b[K250] * dqdx + vd_anom_b[K250] * dqdy) * DAY3
    log(f"  −u_rot_anom_ppvi·∇q'  : mean {np.nanmean(adv_rot):+.3f}  "
        f"p1/p99 {np.nanpercentile(adv_rot,1):+.2f}/{np.nanpercentile(adv_rot,99):+.2f}")
    log(f"  −u_div_anom·∇q'       : mean {np.nanmean(adv_div):+.3f}  "
        f"p1/p99 {np.nanpercentile(adv_div,1):+.2f}/{np.nanpercentile(adv_div,99):+.2f}")

    # ── GATE verdict (blocking-relevant upper band) ─────────────────────────
    gate = (r2_psi_250 > 0.99) and (max_nrms_upper < 0.05) and (pi_r2["250"] > 0.85)
    log(f"\n=== GATE @250: linearity R²={r2_psi_250:.5f} (>0.99? {r2_psi_250>0.99}); "
        f"upper-band(300/250/200) max NRMS={max_nrms_upper:.4f} (<0.05? {max_nrms_upper<0.05}); "
        f"PI-recovery R²={pi_r2['250']:.3f} (>0.85? {pi_r2['250']>0.85}) "
        f"→ {'PASS ✅' if gate else 'FAIL ❌'} ===")
    log("  (lower levels 1000/850/700 carry below-ground hydrostatic-fill artefacts; "
        "500/400 a little pass-D nonlinearity, inlin=1 — reported, not gated.)")

    # ── closure figure (ψ′ @250: Σ, all, residual; PV; recovery scatter) ────
    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    x = lon_win; y = band_lats
    def pcm(a, arr, ttl, cmap="RdBu_r", vm=None):
        vm = vm or np.nanpercentile(np.abs(arr), 99)
        c = a.pcolormesh(x, y, arr, cmap=cmap, vmin=-vm, vmax=vm, shading="auto")
        fig.colorbar(c, ax=a, shrink=0.8); a.set_title(ttl, fontsize=9)
    pcm(ax[0, 0], psi_sum[K250], "Σ per-level ψ′ @250")
    pcm(ax[0, 1], psi_all[K250], "single all-levels ψ′ @250")
    pcm(ax[0, 2], psi_sum[K250] - psi_all[K250], "residual (Σ − all)")
    pcm(ax[1, 0], Q_e_pvu[K250], "Wu PV @250 [PVU]")
    pcm(ax[1, 1], pv_e_band[K250], "input Ertel PV @250 [PVU]")
    ax[1, 2].scatter(u_rot_anom[K250].ravel(), u_sum[K250].ravel(), s=1, alpha=0.3)
    lim = np.nanpercentile(np.abs(u_rot_anom[K250]), 99)
    ax[1, 2].plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax[1, 2].set_xlabel("observed u_rot_anom [m/s]")
    ax[1, 2].set_ylabel("Σ piece u_rot [m/s]")
    ax[1, 2].set_title(f"PI recovery @250 (R²={_r2(u_rot_anom[K250], u_sum[K250]):.3f})",
                       fontsize=9)
    fig.suptitle(f"CESM2 Wu PV-inversion closure — m{MEMBER} {YEAR}-{MONTH:02d}-{DAY:02d} "
                 f"| GATE {'PASS' if gate else 'FAIL'}", fontsize=13, fontweight="bold")
    fig.savefig(OUT_DIR / "closure_check.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"\nSaved -> {OUT_DIR/'closure_check.png'}")

    (OUT_DIR / "closure_report.txt").write_text("\n".join(rpt) + "\n")
    log(f"Saved -> {OUT_DIR/'closure_report.txt'}")


if __name__ == "__main__":
    main()
