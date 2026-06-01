#!/usr/bin/env python
"""Generate all diagnostic figures for the SH-PPVI handbook.

Produces in handbook/figs/ :
    fig_mean_state.png        Step-13 mean fields + closure check
    fig_q_anomaly.png         Step-14 event q_prime per level
    fig_floor_effect.png      Bad-floor vs sign-preserving floor (direct solve)
    fig_per_level_corr.png    Per-level corr(δψ_PPVI, ψ_anom_ERA5)
    fig_pieces_psi.png        δψ per piece at 500 hPa
    fig_pieces_v250.png       δv per piece at 250 hPa (replica of step 15)
    fig_additivity.png        Sum δψ_p vs δψ_total_ppvi vs ψ_anom_ERA5
    fig_residual_history.png  LGMRES residual decay (if available)

All figures are 150 dpi PNGs.  Run from handbook/ with:
    micromamba run -n blocking python scripts/make_figs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ─── paths ────────────────────────────────────────────────────────────────────
ROOT = Path("/net/flood/data2/users/x_yan/pv_inversion")
SH   = ROOT / "sh"
DATA = ROOT / "data" / "sh_ppvi"
OUT  = Path(__file__).resolve().parents[1] / "figs"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SH))

from sh_ppvi.coords    import G, KAPPA, CP, PLEVS_PA, PI_VALS, d_pi_pi, coriolis
from sh_ppvi.operators import laplacian_inv, helmholtz
from sh_ppvi.forward   import ertel_q_sh

PLEVS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]
PROJ  = ccrs.PlateCarree()

# ─── shared helpers ───────────────────────────────────────────────────────────
def _coastlines(ax):
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.4, color="0.3")
    gl = ax.gridlines(draw_labels=False, lw=0.3, color="0.7", ls="--")
    gl.top_labels = gl.right_labels = False


def _nice_vmax(arr):
    p98 = np.nanpercentile(np.abs(arr), 98)
    for v in [0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100]:
        if p98 <= v:
            return v
    return 100


def _save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  → {p.relative_to(OUT.parent)}")
    plt.close(fig)


# ─── data loaders ─────────────────────────────────────────────────────────────
def load_mean():
    return xr.open_dataset(DATA / "mean_state.nc")


def load_pieces():
    return xr.open_dataset(DATA / "piecewise.nc")


# ============================================================================
# FIG 1 — mean state overview
# ============================================================================
def fig_mean_state():
    print("[fig] mean_state")
    ds  = load_mean()
    lat = ds.latitude.values
    lon = ds.longitude.values
    psi = ds.psi_bar.values
    Phi = ds.Phi_bar.values
    q   = ds.q_bar.values
    qp  = ds.q_pred.values

    fig, axes = plt.subplots(
        2, 2, figsize=(13, 8),
        subplot_kw={"projection": PROJ}, constrained_layout=True,
    )
    k500 = PLEVS.index(500)
    k250 = PLEVS.index(250)

    # ψ̄ at 500 hPa
    ax = axes[0, 0]
    v = _nice_vmax(psi[k500]) * 1e7
    c = ax.contourf(lon, lat, psi[k500] / 1e7,
                    levels=np.linspace(-v, v, 21), cmap="RdBu_r",
                    extend="both", transform=PROJ)
    plt.colorbar(c, ax=ax, label=r"$\bar\psi_{500}$ ($10^7$ m$^2$/s)",
                 orientation="horizontal", pad=0.04)
    _coastlines(ax)
    ax.set_title(r"(a) Mean streamfunction $\bar\psi$ @ 500 hPa")

    # Φ̄ at 500 hPa (geopotential height)
    ax = axes[0, 1]
    z500 = Phi[k500] / 9.81
    c = ax.contourf(lon, lat, z500, levels=21, cmap="viridis", transform=PROJ)
    plt.colorbar(c, ax=ax, label="Z (m)", orientation="horizontal", pad=0.04)
    _coastlines(ax)
    ax.set_title(r"(b) Mean geopotential height $\bar Z$ @ 500 hPa")

    # q̄ at 250 hPa
    ax = axes[1, 0]
    v = _nice_vmax(q[k250])
    c = ax.contourf(lon, lat, q[k250], levels=np.linspace(-v, v, 21),
                    cmap="RdBu_r", extend="both", transform=PROJ)
    plt.colorbar(c, ax=ax, label=r"$\bar q$ (PVU)",
                 orientation="horizontal", pad=0.04)
    _coastlines(ax)
    ax.set_title(r"(c) Mean Ertel PV $\bar q$ @ 250 hPa")

    # closure check (q_pred − q_bar) at 250 hPa
    ax = axes[1, 1]
    diff = qp[k250] - q[k250]
    v = max(_nice_vmax(diff), 0.1)
    c = ax.contourf(lon, lat, diff, levels=np.linspace(-v, v, 21),
                    cmap="RdBu_r", extend="both", transform=PROJ)
    plt.colorbar(c, ax=ax, label=r"$q_{\mathrm{pred}}-\bar q$ (PVU)",
                 orientation="horizontal", pad=0.04)
    _coastlines(ax)
    rmse = float(np.sqrt(np.mean((qp - q) ** 2)))
    ax.set_title(rf"(d) PPVI closure check @ 250 hPa  (RMSE={rmse:.3f} PVU)")

    fig.suptitle("Step 13 — Mean-state inversion (2024-12-24 → 2025-01-23 mean)",
                 fontsize=14, fontweight="bold")
    _save(fig, "fig_mean_state.png")


# ============================================================================
# FIG 2 — q_prime per level (event minus mean)
# ============================================================================
def fig_q_anomaly():
    print("[fig] q_anomaly")
    ds = load_pieces()
    lat, lon = ds.latitude.values, ds.longitude.values
    qp = ds.q_prime_sh.values

    fig, axes = plt.subplots(
        2, 5, figsize=(17, 7),
        subplot_kw={"projection": PROJ}, constrained_layout=True,
    )
    for k, ax in enumerate(axes.ravel()):
        v = _nice_vmax(qp[k])
        c = ax.contourf(lon, lat, qp[k], levels=np.linspace(-v, v, 21),
                        cmap="RdBu_r", extend="both", transform=PROJ)
        _coastlines(ax)
        ax.set_title(f"{PLEVS[k]} hPa  ($\\pm${v:.1f} PVU)", fontsize=9)
        plt.colorbar(c, ax=ax, orientation="horizontal", pad=0.08, shrink=0.85)
    fig.suptitle(r"Step 14 — Event PV anomaly $q' = q_{\mathrm{ev}}-\bar q$  "
                 "(2025-01-08 minus climatology)", fontsize=13, fontweight="bold")
    _save(fig, "fig_q_anomaly.png")


# ============================================================================
# FIG 3 — direct-solve floor effect (bad vs sign-preserving)
# ============================================================================
def fig_floor_effect():
    print("[fig] floor_effect")
    dm = load_mean()
    dp = load_pieces()
    lat, lon = dm.latitude.values, dm.longitude.values
    Phi_bar = dm.Phi_bar.values
    qp = dp.q_prime_sh.values

    ones = np.ones_like(Phi_bar)
    coeff = (G * KAPPA * PI_VALS / (PLEVS_PA * CP))[:, None, None] * ones
    A = coeff * d_pi_pi(Phi_bar)
    floor = 1.0e-20

    # BAD: positive-only floor (the bug we fixed)
    A_bad = np.where(A > floor, A, floor)
    rhs_bad = (qp * 1e-6) / A_bad
    rhs_bad[0] = rhs_bad[-1] = 0.0
    dpsi_bad = laplacian_inv(rhs_bad, lat, lon)
    dpsi_bad[0] = dpsi_bad[-1] = 0.0

    # GOOD: sign-preserving floor (current code)
    A_good = np.where(np.abs(A) < floor, floor, A)
    rhs_good = (qp * 1e-6) / A_good
    rhs_good[0] = rhs_good[-1] = 0.0
    dpsi_good = laplacian_inv(rhs_good, lat, lon)
    dpsi_good[0] = dpsi_good[-1] = 0.0

    k_lo = PLEVS.index(925)  # lower piece, affected level
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5),
        subplot_kw={"projection": PROJ}, constrained_layout=True,
    )

    # bad
    ax = axes[0]
    bad = dpsi_bad[k_lo]
    v = np.nanpercentile(np.abs(bad), 98)
    if not np.isfinite(v) or v == 0:
        v = 1
    c = ax.contourf(lon, lat, bad, levels=np.linspace(-v, v, 21),
                    cmap="RdBu_r", extend="both", transform=PROJ)
    plt.colorbar(c, ax=ax, label=r"$\delta\psi$ (m$^2$/s)",
                 orientation="horizontal", pad=0.04)
    _coastlines(ax)
    rms = float(np.sqrt(np.mean(bad ** 2)))
    ax.set_title(f"(a) BAD positive-only floor:  RMS={rms:.2e}")

    # good
    ax = axes[1]
    good = dpsi_good[k_lo]
    v = _nice_vmax(good)
    c = ax.contourf(lon, lat, good, levels=np.linspace(-v, v, 21),
                    cmap="RdBu_r", extend="both", transform=PROJ)
    plt.colorbar(c, ax=ax, label=r"$\delta\psi$ (m$^2$/s)",
                 orientation="horizontal", pad=0.04)
    _coastlines(ax)
    rms = float(np.sqrt(np.mean(good ** 2)))
    ax.set_title(f"(b) GOOD sign-preserving floor:  RMS={rms:.2e}")

    fig.suptitle(r"Effect of the $A$-floor choice on direct-solve $\delta\psi$ @ 925 hPa",
                 fontsize=13, fontweight="bold")
    _save(fig, "fig_floor_effect.png")


# ============================================================================
# FIG 4 — per-level correlation
# ============================================================================
def fig_per_level_corr():
    print("[fig] per_level_corr")
    ds = load_pieces()
    psi_sum = (ds.psi_lower + ds.psi_middle + ds.psi_upper).values
    psi_anom = ds.psi_ev.values - ds.q_bar.values * 0  # placeholder
    # use mean_state for psi_bar
    dm = load_pieces()
    psi_bar = load_mean().psi_bar.values
    psi_anom = ds.psi_ev.values - psi_bar

    corrs = []
    rms_ppvi, rms_era5 = [], []
    for k in range(len(PLEVS)):
        a = psi_sum[k].ravel()
        b = psi_anom[k].ravel()
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            corrs.append(np.nan)
        else:
            corrs.append(np.corrcoef(a, b)[0, 1])
        rms_ppvi.append(float(np.sqrt(np.mean(a ** 2))))
        rms_era5.append(float(np.sqrt(np.mean(b ** 2))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    # corr
    colours = ["tab:red" if c < 0 else "tab:blue" for c in corrs]
    ax1.barh(range(len(PLEVS)), corrs, color=colours, edgecolor="k", lw=0.5)
    ax1.set_yticks(range(len(PLEVS)))
    ax1.set_yticklabels([f"{p} hPa" for p in PLEVS])
    ax1.invert_yaxis()
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_xlim(-1, 1)
    ax1.set_xlabel(r"corr$\;(\sum\delta\psi_p,\;\psi_{\mathrm{ev}}-\bar\psi)$")
    ax1.set_title("Per-level correlation with ERA5 anomaly")
    ax1.grid(axis="x", alpha=0.3)

    # rms
    ax2.plot(rms_era5, range(len(PLEVS)), "o-", label="ERA5 anomaly", lw=2)
    ax2.plot(rms_ppvi, range(len(PLEVS)), "s--", label="PPVI sum", lw=2)
    ax2.set_yticks(range(len(PLEVS)))
    ax2.set_yticklabels([f"{p} hPa" for p in PLEVS])
    ax2.invert_yaxis()
    ax2.set_xlabel(r"RMS of $\psi$ anomaly (m$^2$/s)")
    ax2.set_title("Magnitude comparison per level")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xscale("log")

    fig.suptitle("Direct-solve PPVI vs ERA5 — per-level diagnostics",
                 fontsize=13, fontweight="bold")
    _save(fig, "fig_per_level_corr.png")


# ============================================================================
# FIG 5 — δψ per piece at 500 hPa
# ============================================================================
def fig_pieces_psi():
    print("[fig] pieces_psi")
    ds = load_pieces()
    lat, lon = ds.latitude.values, ds.longitude.values

    pieces = [("lower", "Lower (1000–925 hPa)"),
              ("middle", "Middle (850–700 hPa)"),
              ("upper", "Upper (600–250 hPa)")]
    k = PLEVS.index(500)

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5),
        subplot_kw={"projection": PROJ}, constrained_layout=True,
    )
    for ax, (name, label) in zip(axes, pieces):
        d = ds[f"psi_{name}"].values[k] / 1e7
        v = _nice_vmax(d)
        c = ax.contourf(lon, lat, d, levels=np.linspace(-v, v, 21),
                        cmap="RdBu_r", extend="both", transform=PROJ)
        plt.colorbar(c, ax=ax,
                     label=r"$\delta\psi_{500}$ ($10^7$ m$^2$/s)",
                     orientation="horizontal", pad=0.04)
        _coastlines(ax)
        ax.set_title(label, fontsize=11)

    fig.suptitle(r"Induced streamfunction anomaly $\delta\psi$ @ 500 hPa "
                 r"from each PV piece (direct solve)",
                 fontsize=13, fontweight="bold")
    _save(fig, "fig_pieces_psi.png")


# ============================================================================
# FIG 6 — δv per piece at 250 hPa (handbook replica of fig8)
# ============================================================================
def fig_pieces_v250():
    print("[fig] pieces_v250")
    ds = load_pieces()
    lat, lon = ds.latitude.values, ds.longitude.values
    k = PLEVS.index(250)

    pieces = [("lower", "Lower (1000–925 hPa)"),
              ("middle", "Middle (850–700 hPa)"),
              ("upper", "Upper (600–250 hPa)")]

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5),
        subplot_kw={"projection": PROJ}, constrained_layout=True,
    )
    for ax, (name, label) in zip(axes, pieces):
        d = ds[f"v_{name}"].values[k]
        v = _nice_vmax(d)
        c = ax.contourf(lon, lat, d, levels=np.linspace(-v, v, 21),
                        cmap="RdBu_r", extend="both", transform=PROJ)
        plt.colorbar(c, ax=ax, label=r"$\delta v_{250}$ (m/s)",
                     orientation="horizontal", pad=0.04)
        _coastlines(ax)
        ax.set_title(label, fontsize=11)

    fig.suptitle("Induced meridional wind at 250 hPa per PV piece "
                 "(Davis 2022 Fig. 8 analogue)",
                 fontsize=13, fontweight="bold")
    _save(fig, "fig_pieces_v250.png")


# ============================================================================
# FIG 7 — additivity check
# ============================================================================
def fig_additivity():
    print("[fig] additivity")
    ds = load_pieces()
    lat, lon = ds.latitude.values, ds.longitude.values
    k = PLEVS.index(500)

    psi_sum = (ds.psi_lower + ds.psi_middle + ds.psi_upper).values[k]
    psi_tot = ds.psi_total_ppvi.values[k]
    psi_bar = load_mean().psi_bar.values[k]
    psi_anom = ds.psi_ev.values[k] - psi_bar

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5),
        subplot_kw={"projection": PROJ}, constrained_layout=True,
    )
    vmax = _nice_vmax(np.stack([psi_sum, psi_tot, psi_anom])) / 1e7

    for ax, dat, title in zip(
        axes,
        [psi_sum / 1e7, psi_tot / 1e7, psi_anom / 1e7],
        [r"(a) $\sum_p\delta\psi_p$  (PPVI pieces sum)",
         r"(b) $\delta\psi_{\mathrm{total}}^{\mathrm{PPVI}}$  (full PPVI)",
         r"(c) $\psi_{\mathrm{ev}}-\bar\psi$  (ERA5 anomaly)"]
    ):
        c = ax.contourf(lon, lat, dat,
                        levels=np.linspace(-vmax, vmax, 21),
                        cmap="RdBu_r", extend="both", transform=PROJ)
        plt.colorbar(c, ax=ax, label=r"$\psi$ ($10^7$ m$^2$/s)",
                     orientation="horizontal", pad=0.04)
        _coastlines(ax)
        ax.set_title(title, fontsize=11)

    # metrics
    sc = float(np.sqrt(np.mean((psi_sum - psi_tot) ** 2)) /
               max(np.sqrt(np.mean(psi_tot ** 2)), 1e-10))
    ph = float(np.sqrt(np.mean((psi_sum - psi_anom) ** 2)) /
               max(np.sqrt(np.mean(psi_anom ** 2)), 1e-10))
    cor = float(np.corrcoef(psi_sum.ravel(), psi_anom.ravel())[0, 1])
    fig.suptitle("Additivity check @ 500 hPa  —  "
                 f"self-consistency RMSE/$\\|\\delta\\psi_{{\\mathrm{{tot}}}}\\|$ = "
                 f"{sc*100:.2e}%   ;   physics RMSE = {ph*100:.1f}%   ;   "
                 f"corr = {cor:.3f}",
                 fontsize=12, fontweight="bold")
    _save(fig, "fig_additivity.png")


# ============================================================================
# main
# ============================================================================
if __name__ == "__main__":
    print(f"Writing figures to {OUT}/")
    fig_mean_state()
    fig_q_anomaly()
    fig_floor_effect()
    fig_per_level_corr()
    fig_pieces_psi()
    fig_pieces_v250()
    fig_additivity()
    print("Done.")
