#!/usr/bin/env python
"""
07_plot_fig8.py
Replicate Fig 8 from clim-JCLI-D-22-0674.1 for the 2025-01-08 00Z CA blocking event.

3-panel figure (one panel per vertical piece):
  Panel 1: PV anomalies at 1000–925 hPa  (lower troposphere)
  Panel 2: PV anomalies at 850–700 hPa   (middle troposphere)
  Panel 3: PV anomalies at 600–250 hPa   (upper troposphere)

Each panel:
  - Filled color: 250-hPa PV advection by induced winds (PVU/day), diverging colormap
  - Black arrows: induced wind V' at 250 hPa (quiver, reference vector in corner)
  - Black contours: total 250-hPa PV anomaly q', CI=0.1 PVU, starting ±0.2 PVU

Domain: 10.5°N–85.5°N, −169.5°E to −40.5°E (N-Pacific, matches grid)
Projection: PlateCarree (simple rectangular)
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(CASE_DIR, "data", "ppvi")
FIG_DIR  = os.path.join(CASE_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

# Methods to plot (one 3-panel figure each). Reads the new SH-gradient
# advection NCs written by 06_read_outs.py.
METHODS = ["linear", "nonlinear"]

# ---------------------------------------------------------------------------
# Plotting configuration
# ---------------------------------------------------------------------------
PIECE_TITLES = [
    "(a) PV anomalies: 1000–925 hPa",
    "(b) PV anomalies: 850–700 hPa",
    "(c) PV anomalies: 600–250 hPa",
]
NPIECES = 3

# Colormap for PV advection (diverging: negative = anticyclonic, positive = cyclonic)
CMAP_ADV = "RdBu_r"
# Levels are now auto-scaled per-figure from data percentiles (see main()).

# PV anomaly contour interval (PVU). Davis Fig 8 uses CI ~ 0.5–1.0 PVU at 250 hPa.
# We auto-pick CI from the data percentile (±p95 ≈ 5–7 PVU → CI ≈ 1 PVU).
# CONT_CI / CONT_MAX_ABS / CONT_MIN are now overridden in main() based on data.
CONT_CI       = 1.0    # PVU (fallback)
CONT_MAX_ABS  = 6.0    # PVU (fallback)
CONT_MIN      = 1.0    # PVU (fallback, abs starting value)

# Quiver settings
QUIVER_SKIP  = 4       # plot every N-th gridpoint
QUIVER_WIDTH = 0.003
REF_SPEED    = 20.0    # m/s for reference arrow label
# Cap displayed wind magnitude (m/s). Anything stronger is rescaled to the cap
# so a single noisy upper-piece cell can't dominate the plot. Set to None to
# disable clipping.
WIND_DISPLAY_CAP = 40.0

# Domain for map extent
LON_MIN, LON_MAX = -170.0, -40.0
LAT_MIN, LAT_MAX =   10.0,  85.0


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def _plot_one(method: str):
    """Render the 3-panel Fig-8 replica for one PPVI method."""
    adv_file = os.path.join(OUT_DIR, f"method_{method}_advection.nc")
    print(f"Loading {adv_file} ...")
    if not os.path.exists(adv_file):
        raise FileNotFoundError(f"{adv_file} not found. Run 06_read_outs.py first.")
    ds = xr.open_dataset(adv_file)

    # New schema (from 06_read_outs.py with pvtend.sh_ops):
    #   coords: piece, latitude, longitude
    #   vars:   q_pert_250, true_pv_tend, PVadv, U_250, V_250, U_ind, V_ind
    lats = ds["latitude"].values   # ascending S→N
    lons = ds["longitude"].values  # 0→360°E
    # Shift lons to [-180, 180) so the Pacific/N-America domain is contiguous.
    if lons.max() > 180.0:
        lons_shift = np.where(lons > 180.0, lons - 360.0, lons)
        order = np.argsort(lons_shift)
        lons = lons_shift[order]
    else:
        order = np.arange(lons.size)
    LON2D, LAT2D = np.meshgrid(lons, lats)

    PVadv   = ds["PVadv"].values[:, :, order]    # (3, NY, NX)  PVU/day
    q_anom  = ds["q_pert_250"].values[:, order]  # (NY, NX)     PVU
    U_ind   = ds["U_ind"].values[:, :, order]    # (3, NY, NX)  m/s
    V_ind   = ds["V_ind"].values[:, :, order]    # (3, NY, NX)  m/s

    print(f"  lats: {lats[0]:.1f}N → {lats[-1]:.1f}N, lons: {lons[0]:.1f}E → {lons[-1]:.1f}E")
    print(f"  PVadv range: {np.nanmin(PVadv):.2f} … {np.nanmax(PVadv):.2f} PVU/day")
    print(f"  q_anom range: {np.nanmin(q_anom):.2f} … {np.nanmax(q_anom):.2f} PVU")

    # ---- Smooth piece-3 (upper) induced winds to suppress upper-BC gridpoint noise ----
    try:
        from scipy.ndimage import gaussian_filter
        for ip in range(NPIECES):
            sigma = 1.5 if ip == 2 else 0.0
            if sigma > 0:
                u_smooth = gaussian_filter(np.nan_to_num(U_ind[ip]), sigma=sigma)
                v_smooth = gaussian_filter(np.nan_to_num(V_ind[ip]), sigma=sigma)
                U_ind[ip] = u_smooth
                V_ind[ip] = v_smooth
                # Recompute advection with smoothed winds for piece 3
                from scipy.ndimage import gaussian_filter as _gf
                # We don't have ERA5 q here, but PVadv from file already used unsmoothed
                # winds. Recompute the same product is awkward without q; instead
                # just smooth the advection field itself.
                PVadv[ip] = gaussian_filter(np.nan_to_num(PVadv[ip]), sigma=sigma)
    except Exception as e:
        print(f"  (scipy gaussian_filter unavailable: {e})")

    # ---- Per-panel PV advection vmax (auto from p98 of that panel only) ----
    nice = np.array([0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100])
    panel_vmax = []
    for ip in range(NPIECES):
        v = float(np.nanpercentile(np.abs(PVadv[ip]), 98))
        if v < 0.5:
            v = 1.0
        v = float(nice[np.argmin(np.abs(nice - v))])
        panel_vmax.append(v)
    print(f"  Per-panel PV-adv vmax (PVU/day): {panel_vmax}")

    # ---- Auto-scale PV anomaly contour interval from p95 ----
    p95 = float(np.nanpercentile(np.abs(q_anom), 95))
    target_ci = p95 / 6.0
    nice_ci = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    ci = float(nice_ci[np.argmin(np.abs(nice_ci - target_ci))])
    cmax = float(np.ceil(p95 / ci) * ci)
    POS_LEVS = np.arange(ci, cmax + ci/2, ci)
    NEG_LEVS = -POS_LEVS[::-1]
    print(f"  q_anom |p95|={p95:.2f}; using CI={ci} PVU, levels ±[{ci}..{cmax}]")

    # ---------------------------------------------------------------------------
    # Build figure: 3 rows × 1 column
    # ---------------------------------------------------------------------------
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        3, 1,
        figsize=(10, 16),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )
    fig.suptitle(
        "250-hPa PV Advection Induced by Piecewise PV Anomalies\n"
        f"2025-01-08 00Z  (CA blocking event)  —  method={method}",
        fontsize=13, fontweight="bold",
    )

    for ip, ax in enumerate(axes):
        # ---- Map background ----
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="gray")
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="lightgray")
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="lightgray")

        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="lightgray",
                          alpha=0.8, linestyle="--")
        gl.top_labels    = False
        gl.right_labels  = False
        gl.xlocator = mticker.FixedLocator(range(-170, -30, 20))
        gl.ylocator = mticker.FixedLocator(range(20, 90, 20))
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}

        # ---- Filled contours: PV advection (per-panel vmax) ----
        vmax_p = panel_vmax[ip]
        cbar_levs_p = np.linspace(-vmax_p, vmax_p, 17)
        norm_p = mcolors.BoundaryNorm(cbar_levs_p, plt.cm.RdBu_r.N)
        cf = ax.contourf(
            LON2D, LAT2D, PVadv[ip],
            levels=cbar_levs_p,
            cmap=CMAP_ADV,
            norm=norm_p,
            transform=proj,
            extend="both",
        )

        # ---- Contour lines: total 250-hPa PV anomaly ----
        # Replace NaNs in the contour field with the field mean so the contour
        # algorithm does not draw spurious edge segments around NaN polygons.
        q_for_contour = np.where(np.isnan(q_anom), np.nanmean(q_anom), q_anom)
        # Positive (solid)
        if len(POS_LEVS) > 0 and np.nanmax(q_anom) >= POS_LEVS[0]:
            cs_pos = ax.contour(
                LON2D, LAT2D, q_for_contour,
                levels=POS_LEVS, colors="black",
                linewidths=0.8, linestyles="solid",
                transform=proj,
            )
            ax.clabel(cs_pos, inline=True, fontsize=6, fmt="%.1f")
        # Negative (dashed)
        if len(NEG_LEVS) > 0 and np.nanmin(q_anom) <= NEG_LEVS[-1]:
            cs_neg = ax.contour(
                LON2D, LAT2D, q_for_contour,
                levels=NEG_LEVS, colors="black",
                linewidths=0.8, linestyles="dashed",
                transform=proj,
            )
            ax.clabel(cs_neg, inline=True, fontsize=6, fmt="%.1f")

        # ---- Quiver: induced winds at 250 hPa ----
        sk = QUIVER_SKIP
        # Copy U/V; cap unphysical magnitudes (piece 3 has top-BC noise)
        u_full = U_ind[ip].copy()
        v_full = V_ind[ip].copy()
        # Mask 1-cell halo (zeros from FD stencil)
        u_full[0, :] = u_full[-1, :] = np.nan
        u_full[:, 0] = u_full[:, -1] = np.nan
        v_full[0, :] = v_full[-1, :] = np.nan
        v_full[:, 0] = v_full[:, -1] = np.nan
        if WIND_DISPLAY_CAP is not None:
            speed_full = np.sqrt(u_full**2 + v_full**2)
            scale_factor = np.where(speed_full > WIND_DISPLAY_CAP,
                                     WIND_DISPLAY_CAP / np.maximum(speed_full, 1e-9),
                                     1.0)
            u_full = u_full * scale_factor
            v_full = v_full * scale_factor
            n_capped = int(np.nansum(speed_full > WIND_DISPLAY_CAP))
            if n_capped > 0:
                print(f"    piece {ip+1}: capped {n_capped} cells with |V|>{WIND_DISPLAY_CAP} m/s")

        qlons = LON2D[::sk, ::sk]
        qlats = LAT2D[::sk, ::sk]
        qu    = u_full[::sk, ::sk]
        qv    = v_full[::sk, ::sk]

        # Hide near-zero vectors
        speed = np.sqrt(qu**2 + qv**2)
        small = (speed < 1.0)
        qu = np.where(small, np.nan, qu)
        qv = np.where(small, np.nan, qv)

        Q = ax.quiver(
            qlons, qlats, qu, qv,
            transform=proj,
            color="black",
            scale=REF_SPEED * 25,   # smaller scale = larger arrows
            width=QUIVER_WIDTH,
            headwidth=4,
            headlength=4,
            zorder=5,
        )
        ax.quiverkey(Q, X=0.90, Y=-0.06, U=REF_SPEED,
                     label=f"{REF_SPEED:.0f} m/s",
                     labelpos="E", fontproperties={"size": 8})

        # ---- Title and colorbar ----
        ax.set_title(PIECE_TITLES[ip], fontsize=10, loc="left", pad=3)
        plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.01,
                     label="PV advection (PVU day⁻¹)", fraction=0.03)

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    png_path = os.path.join(FIG_DIR, f"fig8_3panel_{method}.png")
    pdf_path = os.path.join(FIG_DIR, f"fig8_3panel_{method}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")
    plt.close(fig)


if __name__ == "__main__":
    for m in METHODS:
        print(f"\n--- 3-panel plot: method = {m} ---")
        _plot_one(m)
    print("\n=== 07_plot_fig8.py complete ===")
