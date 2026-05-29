#!/usr/bin/env python
"""
08_plot_2x2.py — 2×2 Davis Fig-8 replica per balance method.

For each method (linear, nonlinear) builds a 2×2 figure:
  (a) lower piece (850/700 hPa)  PV advection + induced 250-hPa winds
  (b) mid piece (500/400 hPa)    same
  (c) upper piece (300/250 hPa)  same  (Gaussian smoothed to suppress top-BC noise)
  (d) sum vs. true tendency      shading = true ERA5 tendency, contour = piece sum

Fixes ported from scripts/07_plot_fig8.py:
  • per-panel auto-vmax from p98, snapped to nice values
  • Gaussian-smooth upper-piece PVadv and U/V (sigma=1.5)
  • cap displayed wind magnitude at 40 m/s
  • mask 1-cell halo in induced winds
  • constrained_layout=True (tight_layout breaks with Cartopy)
  • per-panel colorbar (different scales per piece)

Inputs : data/ppvi/method_{linear,nonlinear}_advection.nc
Outputs: figs/fig8_method_{linear,nonlinear}.{png,pdf}

Run with: micromamba run -n fourcastnetv2 python scripts/08_plot_2x2.py
"""

from __future__ import annotations
import os, warnings
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    warnings.warn("cartopy not found — using plain axes")

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not found — upper-piece smoothing disabled")

# ── paths ─────────────────────────────────────────────────────────────────────
CASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PPVI_DIR = os.path.join(CASE_DIR, "data", "ppvi")
FIG_DIR  = os.path.join(CASE_DIR, "figs");  os.makedirs(FIG_DIR, exist_ok=True)

# ── crop domain (NE Pacific + North America) ──────────────────────────────────
LAT_MIN, LAT_MAX = 10.5, 85.5
LON_MIN, LON_MAX = 190.5, 319.5
# Mask polar cap for display — the 1/cos(lat) factor in ∂q/∂x produces noise
# poleward of POLAR_MASK_LAT that dominates the p98-based vmax. Mask BEFORE
# colour-scale auto-calc and BEFORE rendering.
POLAR_MASK_LAT = 88.0  # SH operators are pole-stable; mask only the literal pole

CMAP             = "RdBu_r"
QUIV_SKIP        = 4
REF_SPEED        = 20.0
QUIV_WIDTH       = 0.003
WIND_DISPLAY_CAP = 40.0       # m/s
UPPER_SMOOTH_SIGMA = 1.5      # Gaussian sigma for upper-piece smoothing
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]
PIECE_TITLES = [
    "Lower (850/700 hPa)",
    "Middle (500/400 hPa)",
    "Upper (300/250 hPa)",
    "Sum vs. true tendency",
]

# Snap-to-nice grid for per-panel vmax
NICE_VMAX = np.array([0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100])


# ─────────────────────────────────────────────────────────────────────────────
def _crop(arr, lat, lon):
    """Crop 2-D or 3-D (N,lat,lon) array to case domain."""
    la_idx = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]
    lo_idx = np.where((lon >= LON_MIN) & (lon <= LON_MAX))[0]
    if arr.ndim == 2:
        return arr[np.ix_(la_idx, lo_idx)], lat[la_idx], lon[lo_idx]
    return arr[:, la_idx, :][:, :, lo_idx], lat[la_idx], lon[lo_idx]


def _nice_vmax(field):
    """Pick a nice symmetric vmax for diverging shading from p98 of |field|."""
    v = float(np.nanpercentile(np.abs(field), 98))
    if not np.isfinite(v) or v < 0.5:
        v = 1.0
    return float(NICE_VMAX[np.argmin(np.abs(NICE_VMAX - v))])


def _smooth_upper(PVadv, U_ind, V_ind):
    """Gaussian-smooth upper-piece (index 2) PVadv + induced winds."""
    if not HAS_SCIPY:
        return PVadv, U_ind, V_ind
    PVadv = PVadv.copy(); U_ind = U_ind.copy(); V_ind = V_ind.copy()
    s = UPPER_SMOOTH_SIGMA
    PVadv[2] = gaussian_filter(np.nan_to_num(PVadv[2]), sigma=s)
    U_ind[2] = gaussian_filter(np.nan_to_num(U_ind[2]), sigma=s)
    V_ind[2] = gaussian_filter(np.nan_to_num(V_ind[2]), sigma=s)
    return PVadv, U_ind, V_ind


def _cap_winds(u, v):
    """Mask 1-cell halo + cap magnitudes at WIND_DISPLAY_CAP m/s."""
    u = u.copy(); v = v.copy()
    u[0, :] = u[-1, :] = np.nan
    u[:, 0] = u[:, -1] = np.nan
    v[0, :] = v[-1, :] = np.nan
    v[:, 0] = v[:, -1] = np.nan
    if WIND_DISPLAY_CAP is not None:
        speed = np.sqrt(u**2 + v**2)
        scale = np.where(speed > WIND_DISPLAY_CAP,
                         WIND_DISPLAY_CAP / np.maximum(speed, 1e-9), 1.0)
        u = u * scale;  v = v * scale
    return u, v


def _setup_map(ax, lon_plot, lat):
    if not HAS_CARTOPY:
        return None
    proj = ccrs.PlateCarree()
    ax.set_extent([float(lon_plot.min()), float(lon_plot.max()),
                   float(lat.min()),     float(lat.max())], crs=proj)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="gray")
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="lightgray",
                   linestyle=":")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                      alpha=0.5, linestyle="--")
    gl.top_labels = gl.right_labels = False
    gl.xlocator   = mticker.FixedLocator(np.arange(-180, 181, 30))
    gl.ylocator   = mticker.FixedLocator(np.arange(0, 91, 20))
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    return proj


def _render_piece(ax, pv, u_ind, v_ind, lat, lon,
                  title, panel_label):
    # roll lon 190..319 -> -170..-40 for cartopy display
    lon360 = np.where(lon > 180, lon - 360.0, lon)
    order  = np.argsort(lon360)
    lon_p  = lon360[order]
    pv_p   = pv[:, order]
    u_p    = u_ind[:, order]
    v_p    = v_ind[:, order]

    proj = _setup_map(ax, lon_p, lat)
    transform = proj if proj is not None else None
    kw = dict(transform=transform) if transform is not None else {}

    XX, YY = np.meshgrid(lon_p, lat)

    vmax = _nice_vmax(pv)
    bounds = np.linspace(-vmax, vmax, 17)
    norm   = mcolors.BoundaryNorm(bounds, matplotlib.colormaps[CMAP].N)
    im = ax.contourf(XX, YY, pv_p, levels=bounds, cmap=CMAP, norm=norm,
                     extend="both", **kw)

    u_disp, v_disp = _cap_winds(u_p, v_p)
    sk = QUIV_SKIP
    speed = np.sqrt(u_disp**2 + v_disp**2)
    u_disp = np.where(speed < 1.0, np.nan, u_disp)
    v_disp = np.where(speed < 1.0, np.nan, v_disp)
    Q = ax.quiver(XX[::sk, ::sk], YY[::sk, ::sk],
                  u_disp[::sk, ::sk], v_disp[::sk, ::sk],
                  scale=REF_SPEED * 25, width=QUIV_WIDTH, color="k",
                  headwidth=4, headlength=4, zorder=5, **kw)
    ax.quiverkey(Q, X=0.90, Y=-0.08, U=REF_SPEED,
                 label=f"{REF_SPEED:.0f} m/s", labelpos="E",
                 fontproperties={"size": 7})

    ax.set_title(f"{panel_label}  {title}  (|max|={vmax:g} PVU/day)",
                 fontsize=9, loc="left")
    cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02,
                      fraction=0.04)
    cb.set_label(r"PVU day$^{-1}$", fontsize=8)
    cb.ax.tick_params(labelsize=6)


def _render_sum(ax, true_tend, pv_sum, lat, lon, title, panel_label):
    lon360 = np.where(lon > 180, lon - 360.0, lon)
    order  = np.argsort(lon360)
    lon_p  = lon360[order]
    true_p = true_tend[:, order]
    sum_p  = pv_sum[:, order]

    proj = _setup_map(ax, lon_p, lat)
    transform = proj if proj is not None else None
    kw = dict(transform=transform) if transform is not None else {}

    XX, YY = np.meshgrid(lon_p, lat)

    vmax = max(_nice_vmax(true_tend), _nice_vmax(pv_sum))
    bounds = np.linspace(-vmax, vmax, 17)
    norm   = mcolors.BoundaryNorm(bounds, matplotlib.colormaps[CMAP].N)
    im = ax.contourf(XX, YY, true_p, levels=bounds, cmap=CMAP, norm=norm,
                     extend="both", **kw)

    # contour piece-sum (use ~8 contour levels at fractions of vmax)
    clev_step = vmax / 4.0
    clev = np.arange(clev_step, vmax + clev_step/2, clev_step)
    clev = np.concatenate([-clev[::-1], clev])
    cs_pos = ax.contour(XX, YY, sum_p,
                        levels=clev[clev > 0],
                        colors="k", linewidths=0.7, linestyles="solid", **kw)
    cs_neg = ax.contour(XX, YY, sum_p,
                        levels=clev[clev < 0],
                        colors="k", linewidths=0.7, linestyles="dashed", **kw)
    try:
        ax.clabel(cs_pos, inline=True, fontsize=5, fmt="%.1f")
        ax.clabel(cs_neg, inline=True, fontsize=5, fmt="%.1f")
    except Exception:
        pass

    ax.set_title(f"{panel_label}  {title}  (shade=true, contour=Σpieces)",
                 fontsize=9, loc="left")
    cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02,
                      fraction=0.04)
    cb.set_label(r"PVU day$^{-1}$ (true)", fontsize=8)
    cb.ax.tick_params(labelsize=6)


# ─────────────────────────────────────────────────────────────────────────────
def _build_figure(method: str) -> None:
    adv_nc = os.path.join(PPVI_DIR, f"method_{method}_advection.nc")
    if not os.path.exists(adv_nc):
        print(f"  [SKIP] {adv_nc} not found");  return

    ds = xr.open_dataset(adv_nc)
    lat = ds["latitude"].values.astype(float)
    lon = ds["longitude"].values.astype(float)

    PVadv     = ds["PVadv"].values.astype(float)
    U_ind     = ds["U_ind"].values.astype(float)
    V_ind     = ds["V_ind"].values.astype(float)
    true_tend = ds["true_pv_tend"].values.astype(float)
    pv_sum    = PVadv.sum(axis=0)

    # Smooth upper piece BEFORE cropping (to use full-domain neighbourhood)
    PVadv, U_ind, V_ind = _smooth_upper(PVadv, U_ind, V_ind)
    pv_sum = PVadv.sum(axis=0)        # rebuild sum after smoothing piece 2

    PVadv,     crop_lat, crop_lon = _crop(PVadv,     lat, lon)
    U_ind,     _, _               = _crop(U_ind,     lat, lon)
    V_ind,     _, _               = _crop(V_ind,     lat, lon)
    true_tend, _, _               = _crop(true_tend, lat, lon)
    pv_sum,    _, _               = _crop(pv_sum,    lat, lon)

    # Polar mask: blank out lat > POLAR_MASK_LAT and lat < 15 (edge noise from
    # 1/cos(lat) gradient near pole and equator). Applied to PV-advection /
    # tendency shading and to induced-wind quivers.
    polar_mask = (crop_lat > POLAR_MASK_LAT) | (crop_lat < 15.0)
    if polar_mask.any():
        PVadv[:, polar_mask, :]     = np.nan
        U_ind[:, polar_mask, :]     = np.nan
        V_ind[:, polar_mask, :]     = np.nan
        true_tend[polar_mask, :]    = np.nan
        pv_sum[polar_mask, :]       = np.nan

    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig, axes = plt.subplots(2, 2, figsize=(14, 9),
                                 subplot_kw={"projection": proj},
                                 constrained_layout=True)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9),
                                 constrained_layout=True)
    axes_flat = axes.flatten()

    for i in range(3):
        _render_piece(axes_flat[i], PVadv[i], U_ind[i], V_ind[i],
                      crop_lat, crop_lon,
                      title=PIECE_TITLES[i], panel_label=PANEL_LABELS[i])
    _render_sum(axes_flat[3], true_tend, pv_sum, crop_lat, crop_lon,
                title=PIECE_TITLES[3], panel_label=PANEL_LABELS[3])

    fig.suptitle(
        f"PPVI-induced 250 hPa PV advection — 2025-01-08 00Z CA blocking  "
        f"(method: {method})", fontsize=11)

    for ext in ("png", "pdf"):
        out = os.path.join(FIG_DIR, f"fig8_method_{method}.{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 60)
    print("08_plot_2x2.py — 2×2 Fig-8 replica")
    print("=" * 60)
    for m in ("linear", "nonlinear"):
        print(f"\n--- method = {m} ---")
        _build_figure(m)
    print("\n08_plot_2x2.py complete.")
