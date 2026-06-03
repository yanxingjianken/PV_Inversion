# %% [markdown]
# Step 11 — Plot B: Upper-Level (250 hPa) Streamfunction Induced by Each PV Piece
#
# 3-panel map showing the 250-hPa streamfunction perturbation ψ′ induced by
# each of the three vertical PV pieces (lower / middle / upper).
#
# **Each panel (one per inducing piece):**
# - **Filled shading**: ψ′ at 250 hPa induced by that piece's PV anomaly (RdBu_r)
# - **Black solid contours**: total 250-hPa PV anomaly (same in all panels)
# - **Black dashed contours**: mean PV anomaly over the inducing piece's levels
#
# Reads: `piecewise_psi.nc` + ERA5 `mean_clim.nc` / `event.nc`
#
# %% [markdown]
# ## 1. Load Data
#
# %%
import numpy as np, xarray as xr, matplotlib.pyplot as plt
import matplotlib.colors as mcolors, matplotlib.ticker as mticker
import cartopy.crs as ccrs, cartopy.feature as cfeature
from pathlib import Path
import yaml

import sys
_sys_path_root = str(Path(__file__).resolve().parent.parent.parent)
if _sys_path_root not in sys.path:
    sys.path.insert(0, _sys_path_root)
import config

STEP_DIR = Path(__file__).resolve().parent
OUT_DIR = Path(config.WU_OUT_DIR)
CLIM_DIR = Path(config.CLIM_DIR)

_YAML_CFG_PATH = Path(_sys_path_root) / "wu_config.yaml"
with open(_YAML_CFG_PATH) as _f:
    _yaml_cfg = yaml.safe_load(_f)
_pieces = _yaml_cfg["pieces"]

NPIECES = 3
K250 = 6  # 250 hPa is index 6 of 8 [0=1000,1=850,2=700,3=500,4=400,5=300,6=250,7=200]

# ── Load ψ (= streamfunction in m²/s) from piecewise_psi.nc ──
ds_psi = xr.open_dataset(OUT_DIR / "piecewise_psi.nc")
PSI_pieces = ds_psi["psi"].values  # (piece, plev, lat, lon); ψ in m²/s
plevs = ds_psi["plev"].values
lats_arr = ds_psi["lat"].values
lons = ds_psi["lon"].values

# ── Compute PV anomaly at all levels from clim data ──
ds_mean = xr.open_dataset(CLIM_DIR / "mean_clim.nc")
ds_event = xr.open_dataset(CLIM_DIR / "event.nc")

LON2D, LAT2D = np.meshgrid(lons, lats_arr)


def ertel_pv_3d(t3d, u3d, v3d, plev_Pa, lat_arr):
    """Ertel PV in SI: K·m²·kg⁻¹·s⁻¹. Returns (plev, lat, lon)."""
    P0, Rd, Cp = 1.0e5, 287.0, 1004.0
    theta = t3d * (P0 / plev_Pa[:, None, None]) ** (Rd / Cp)
    dtheta_dp = np.gradient(theta, plev_Pa, axis=0)

    R_E = 6.371e6
    dlat_rad = np.deg2rad(np.gradient(lat_arr))[None, :, None]
    dlon_rad = np.deg2rad(np.gradient(lons))[None, None, :]
    coslat = np.cos(np.deg2rad(lat_arr))[None, :, None]

    dudy = np.gradient(u3d, axis=1) / (R_E * dlat_rad)
    dvdx = np.gradient(v3d, axis=2) / (R_E * coslat * dlon_rad)
    zeta = dvdx - dudy
    f = 2 * 7.292e-5 * np.sin(np.deg2rad(lat_arr))[None, :, None]
    g = 9.81
    pv = -g * (zeta + f) * dtheta_dp  # SI — NO *1e6
    return pv


plev_Pa = ds_mean.pressure_level.values.astype(float) * 100.0

PV_mean = ertel_pv_3d(
    ds_mean["t"].values, ds_mean["u"].values, ds_mean["v"].values,
    plev_Pa, lats_arr)
PV_event = ertel_pv_3d(
    ds_event["t"].values, ds_event["u"].values, ds_event["v"].values,
    plev_Pa, lats_arr)
PV_anom = PV_event - PV_mean  # (8, 51, 87)

# Total 250 hPa PV anomaly (solid contours, same in every panel) — in SI
q_anom_250 = PV_anom[K250].copy()
PV_SCALE = 1e6  # for display printout

print(f"PV anomaly @250 hPa range: [{q_anom_250.min()*PV_SCALE:.2f}, {q_anom_250.max()*PV_SCALE:.2f}] PVU  "
      f"(SI: [{q_anom_250.min():.2e}, {q_anom_250.max():.2e}])")

# %% [markdown]
# ## 2. Build per-Piece Arrays
#
# %%
# Piece names & level indices from YAML (Fortran 1-indexed K → Python 0-indexed k)
piece_names = list(_pieces.keys())  # ["lower", "middle", "upper"]
piece_k_idx = {
    pname: [k - 1 for k in _pieces[pname]["levels"]]
    for pname in piece_names
}
piece_labels = {
    pname: f"{pname.capitalize()} ({_pieces[pname]['hpa']} hPa)"
    for pname in piece_names
}
# Reorder for display: lower → middle → upper (left to right)
display_order = ["lower", "middle", "upper"]

# Induced ψ′ at 250 hPa (already in m²/s from .nc)
psi_250 = {
    pname: PSI_pieces[i, K250]
    for i, pname in enumerate(piece_names)
}

# Mean PV anomaly over each piece's levels (dashed contours)
pv_mean_piece = {}
for pname in piece_names:
    km = piece_k_idx[pname]
    pv_mean_piece[pname] = np.mean(PV_anom[km], axis=0)

print("Induced ψ′ @250 hPa:")
for pname in display_order:
    p = psi_250[pname]
    print(f"  {pname}: ψ′ range [{p.min():.2e}, {p.max():.2e}] m²/s")
print("Mean PV anomaly per piece (PVU / SI):")
for pname in display_order:
    p = pv_mean_piece[pname]
    print(f"  {pname}: PV anom range [{p.min()*PV_SCALE:.2f}, {p.max()*PV_SCALE:.2f}] PVU  "
          f"(SI: [{p.min():.2e}, {p.max():.2e}])")

# %% [markdown]
# ## 3. Auto-Scale Configuration
#
# %%
# ---- Per-panel ψ′ vmax (nice numbers) ----
nice = np.array([0.5e6, 1e6, 2e6, 3e6, 5e6, 7.5e6, 1e7, 1.5e7, 2e7, 3e7, 5e7])
panel_vmax = []
for pname in display_order:
    v = float(np.nanpercentile(np.abs(psi_250[pname]), 98))
    if not np.isfinite(v) or v <= 0:
        v = 1e6
    v = float(nice[np.argmin(np.abs(nice - v))])
    panel_vmax.append(v)
print(f"Per-panel ψ′ vmax: {panel_vmax}")

# ---- PV anomaly contour levels (from total 250 hPa) — SI-scale ---
p95_q = float(np.nanpercentile(np.abs(q_anom_250), 95))
nice_ci = np.array([1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5])
ci = float(nice_ci[np.argmin(np.abs(nice_ci - p95_q / 6))])
POS_LEVS = np.arange(ci, np.nanmax(q_anom_250) + ci / 2, ci)
NEG_LEVS = -POS_LEVS[::-1]
# Filter empty levels
if len(POS_LEVS) > 0 and POS_LEVS[0] > np.nanmax(q_anom_250):
    POS_LEVS = np.array([])
if len(NEG_LEVS) > 0 and NEG_LEVS[-1] < np.nanmin(q_anom_250):
    NEG_LEVS = np.array([])
print(f"Total 250 hPa PV anom |p95| = {p95_q*PV_SCALE:.2f} PVU (SI: {p95_q:.2e}); CI = {ci*PV_SCALE:.1e} PVU")

# ---- Dashed contour levels (inducing piece mean PV) ----
# Use same CI as solid but per-piece range
dash_levels = {}
for pname in display_order:
    pv_piece = pv_mean_piece[pname]
    pmax = np.nanmax(np.abs(pv_piece))
    if pmax < ci:
        dash_levels[pname] = (np.array([]), np.array([]))
    else:
        dpos = np.arange(ci, pmax + ci / 2, ci)
        dneg = -dpos[::-1]
        dash_levels[pname] = (dpos, dneg)

# %% [markdown]
# ## 4. Plot — 3-Panel ψ′ Induced at 250 hPa by Each PV Piece
#
# %%
proj = ccrs.PlateCarree(central_longitude=200)  # center on Pacific domain (no 180° seam)
pc = ccrs.PlateCarree()
fig, axes = plt.subplots(
    1, 3, figsize=(24, 8),
    subplot_kw={"projection": proj}
)
fig.suptitle(
    "250-hPa Streamfunction Induced by Piecewise PV Anomalies\n"
    "CA Blocking Event 2025-01-08 00Z  |  INLIN=0  |  8 Levels  |  30yr Clim",
    fontsize=13, fontweight="bold")

for ip, (ax, pname) in enumerate(zip(axes, display_order)):
    # --- Map background ---
    ax.set_extent([config.LON_W - 2, config.LON_E + 2,
                   config.LAT_S - 2, config.LAT_N + 2], crs=pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.7, edgecolor="gray")
    ax.add_feature(cfeature.BORDERS, lw=0.4, edgecolor="lightgray")
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.3, edgecolor="lightgray")
    gl = ax.gridlines(draw_labels=(ip == 0), lw=0.4, color="lightgray",
                      alpha=0.8, ls="--")
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(-170, -30, 20))
    gl.ylocator = mticker.FixedLocator(range(20, 90, 20))
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    # --- Filled: induced ψ′ at 250 hPa ---
    vmax_p = panel_vmax[ip]
    cbar_levs = np.linspace(-vmax_p, vmax_p, 17)
    norm = mcolors.BoundaryNorm(cbar_levs, plt.cm.RdBu_r.N)
    psi_field = psi_250[pname]
    cf = ax.contourf(LON2D, LAT2D, psi_field, levels=cbar_levs,
                     cmap="RdBu_r", norm=norm, transform=pc, extend="both")
    cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.08,
                        label="ψ′ [m²/s]", fraction=0.05, shrink=0.85)
    cbar.ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")

    # --- Black SOLID: total 250 hPa PV anomaly ---
    if len(POS_LEVS) > 0:
        cs_p = ax.contour(LON2D, LAT2D, q_anom_250, levels=POS_LEVS,
                          colors="black", linewidths=0.8, linestyles="solid",
                          transform=pc)
        ax.clabel(cs_p, inline=True, fontsize=6, fmt="%.1f")
    if len(NEG_LEVS) > 0:
        cs_n = ax.contour(LON2D, LAT2D, q_anom_250, levels=NEG_LEVS,
                          colors="black", linewidths=0.8, linestyles="dashed",
                          transform=pc)
        ax.clabel(cs_n, inline=True, fontsize=6, fmt="%.1f")

    # --- Black DASHED: inducing piece mean PV anomaly ---
    dpos, dneg = dash_levels[pname]
    if len(dpos) > 0:
        ax.contour(LON2D, LAT2D, pv_mean_piece[pname], levels=dpos,
                   colors="black", linewidths=1.2, linestyles="dashed",
                   transform=pc)
    if len(dneg) > 0:
        ax.contour(LON2D, LAT2D, pv_mean_piece[pname], levels=dneg,
                   colors="black", linewidths=1.2, linestyles="dashed",
                   transform=pc)

    ax.set_title(piece_labels[pname], fontsize=11, fontweight="bold")

# Legend annotation on first panel
axes[0].text(0.02, 0.02,
             "solid = 250 hPa PV anom (all panels)\n"
             "dashed = inducing-piece mean PV anom",
             transform=axes[0].transAxes, ha="left", va="bottom",
             fontsize=7, color="black", style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

plt.tight_layout(rect=[0, 0, 1, 0.94])

fname_png = STEP_DIR / "upper_psi_by_piece_250hpa.png"
fname_pdf = STEP_DIR / "upper_psi_by_piece_250hpa.pdf"
fig.savefig(fname_png, dpi=200, bbox_inches="tight")
fig.savefig(fname_pdf, bbox_inches="tight")
print(f"\n✓ Saved: {fname_png}")
print(f"✓ Saved: {fname_pdf}")
plt.close(fig)

print("\n=== Plot 1 complete — upper_psi_by_piece ===")
