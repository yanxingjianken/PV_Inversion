# %% [markdown]
# Step 10 — Fig 8 Replica: 3-Panel Davis-Style PV Advection Plot
#
# Replicates Davis et al. (2022, J. Climate) Fig 8 for the
# 2025-01-08 00Z California blocking event.
#
# **Each panel (one per vertical PV piece)**:
# - **Filled colour**: 250-hPa PV advection by induced winds (RdBu_r, per-panel auto-scaled)
# - **Black solid contours**: positive total 250-hPa PV anomaly
# - **Black dashed contours**: negative PV anomaly
# - **Black arrows**: induced wind vectors at 250 hPa (capped at 40 m/s; ref arrow 20 m/s)
#
# **Pieces** (8 levels, no interp, INLIN=0):
# - Piece 1 (lower):  1000–850 hPa
# - Piece 2 (middle): 700–400 hPa
# - Piece 3 (upper):  300–200 hPa
#
# %% [markdown]
# ## 1. Load & Prepare Data
#
# %%
import numpy as np, xarray as xr, matplotlib.pyplot as plt
import matplotlib.colors as mcolors, matplotlib.ticker as mticker
import cartopy.crs as ccrs, cartopy.feature as cfeature
from pathlib import Path
from scipy.ndimage import gaussian_filter

import sys; from pathlib import Path as _Path
_sys_path_root = str(_Path(__file__).resolve().parent.parent.parent)
if _sys_path_root not in sys.path: sys.path.insert(0, _sys_path_root)
import config
STEP_DIR = _Path(__file__).resolve().parent
OUT_DIR = _Path(config.WU_OUT_DIR)

ds = xr.open_dataset(OUT_DIR / "pv_advection.nc")
lats = ds.lat.values; lons = ds.lon.values
LON2D, LAT2D = np.meshgrid(lons, lats)

# Read SI fields from .nc; scale to display-friendly units for plotting
PV_SCALE = 1e6           # SI → PVU
PVADV_DAY_SCALE = 1e6 * 86400.0  # SI s⁻¹ → PVU/day
PVadv   = ds.PVadv.values.copy() * PVADV_DAY_SCALE   # (3, 51, 87)  PVU/day
q_anom  = ds.Q_anom_250.values * PV_SCALE             # (51, 87)     PV anomaly [PVU]
U_ind   = ds.U_induced_250.values.copy()
V_ind   = ds.V_induced_250.values.copy()

NPIECES = 3

# %% [markdown]
# ## 2. Auto-Scale Configuration
#
# %%
# ---- Smooth piece 3 ----
for ip in range(NPIECES):
    if ip == 2:
        sigma = 1.5
        U_ind[ip] = gaussian_filter(np.nan_to_num(U_ind[ip]), sigma=sigma)
        V_ind[ip] = gaussian_filter(np.nan_to_num(V_ind[ip]), sigma=sigma)
        PVadv[ip] = gaussian_filter(np.nan_to_num(PVadv[ip]), sigma=sigma)

# ---- Per-panel PV advection vmax ----
nice = np.array([0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100])
panel_vmax = []
for ip in range(NPIECES):
    v = float(np.nanpercentile(np.abs(PVadv[ip]), 98))
    if v < 0.5: v = 1.0
    v = float(nice[np.argmin(np.abs(nice - v))])
    panel_vmax.append(v)
print(f"Per-panel PV adv vmax [PVU/day]: {panel_vmax}")

# ---- Auto-scale q_anom contour interval ----
p95 = float(np.nanpercentile(np.abs(q_anom), 95))
nice_ci = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
ci = float(nice_ci[np.argmin(np.abs(nice_ci - p95/6))])
cmax = float(np.ceil(p95 / ci) * ci)
POS_LEVS = np.arange(ci, cmax + ci/2, ci)
NEG_LEVS = -POS_LEVS[::-1]
print(f"q_anom |p95| = {p95:.2f} PVU; CI = {ci} PVU, levels ±[{ci}..{cmax}]")

# ---- Wind display cap ----
WIND_CAP = 40.0
for ip in range(NPIECES):
    speed = np.sqrt(U_ind[ip]**2 + V_ind[ip]**2)
    sf = np.where(speed > WIND_CAP, WIND_CAP / np.maximum(speed, 1e-9), 1.0)
    n_capped = int(np.nansum(speed > WIND_CAP))
    if n_capped > 0:
        print(f"  Piece {ip+1}: capped {n_capped} wind cells > {WIND_CAP} m/s")
    U_ind[ip] *= sf; V_ind[ip] *= sf

# ---- Mask boundary halos ----
for ip in range(NPIECES):
    U_ind[ip, 0, :] = U_ind[ip, -1, :] = np.nan
    U_ind[ip, :, 0] = U_ind[ip, :, -1] = np.nan
    V_ind[ip, 0, :] = V_ind[ip, -1, :] = np.nan
    V_ind[ip, :, 0] = V_ind[ip, :, -1] = np.nan
    PVadv[ip, 0, :] = PVadv[ip, -1, :] = np.nan
    PVadv[ip, :, 0] = PVadv[ip, :, -1] = np.nan

# %% [markdown]
# ## 3. Plot — 3-Panel Davis Fig 8 Replica
#
# %%
proj = ccrs.PlateCarree()
# Load piece names from YAML for consistency
import yaml
_YAML_CFG_PATH = _Path(_sys_path_root) / "wu_config.yaml"
with open(_YAML_CFG_PATH) as _f:
    _yaml_cfg = yaml.safe_load(_f)
_pieces = _yaml_cfg["pieces"]

piece_titles = [
    f"(a) PV anomalies: {_pieces['lower']['hpa']} hPa",
    f"(b) PV anomalies: {_pieces['middle']['hpa']} hPa",
    f"(c) PV anomalies: {_pieces['upper']['hpa']} hPa",
]
QUIVER_SKIP = 4
REF_SPEED = 20.0

# NaN-safe contour field
q_contour = np.where(np.isnan(q_anom), np.nanmean(q_anom), q_anom)

fig, axes = plt.subplots(3, 1, figsize=(10, 16),
                         subplot_kw={"projection": proj})
fig.suptitle("250-hPa PV Advection Induced by Piecewise PV Anomalies\n"
             "2025-01-08 00Z  |  INLIN=0  |  8 Levels  |  30yr Clim",
             fontsize=13, fontweight="bold")

for ip, ax in enumerate(axes):
    # Map
    ax.set_extent([-170, -40, 10, 85], crs=proj)
    ax.add_feature(cfeature.COASTLINE, lw=0.7, edgecolor="gray")
    ax.add_feature(cfeature.BORDERS, lw=0.4, edgecolor="lightgray")
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.3, edgecolor="lightgray")
    gl = ax.gridlines(draw_labels=True, lw=0.4, color="lightgray", alpha=0.8, ls="--")
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(-170, -30, 20))
    gl.ylocator = mticker.FixedLocator(range(20, 90, 20))
    gl.xlabel_style = {"size": 8}; gl.ylabel_style = {"size": 8}

    # Filled PV advection (per-panel vmax)
    vmax_p = panel_vmax[ip]
    cbar_levs = np.linspace(-vmax_p, vmax_p, 17)
    norm = mcolors.BoundaryNorm(cbar_levs, plt.cm.RdBu_r.N)
    cf = ax.contourf(LON2D, LAT2D, PVadv[ip], levels=cbar_levs,
                     cmap="RdBu_r", norm=norm, transform=proj, extend="both")
    plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.01,
                 label="PV advection (PVU day⁻¹)", fraction=0.03)

    # PV anomaly contours
    if len(POS_LEVS) > 0 and np.nanmax(q_anom) >= POS_LEVS[0]:
        cs_p = ax.contour(LON2D, LAT2D, q_contour, levels=POS_LEVS,
                          colors="black", linewidths=0.8, linestyles="solid", transform=proj)
        ax.clabel(cs_p, inline=True, fontsize=6, fmt="%.1f")
    if len(NEG_LEVS) > 0 and np.nanmin(q_anom) <= NEG_LEVS[-1]:
        cs_n = ax.contour(LON2D, LAT2D, q_contour, levels=NEG_LEVS,
                          colors="black", linewidths=0.8, linestyles="dashed", transform=proj)
        ax.clabel(cs_n, inline=True, fontsize=6, fmt="%.1f")

    # Wind quivers
    sk = QUIVER_SKIP
    qlons, qlats = LON2D[::sk, ::sk], LAT2D[::sk, ::sk]
    qu, qv = U_ind[ip, ::sk, ::sk], V_ind[ip, ::sk, ::sk]
    speed = np.sqrt(qu**2 + qv**2)
    qu = np.where(speed < 1.0, np.nan, qu)
    qv = np.where(speed < 1.0, np.nan, qv)

    Q = ax.quiver(qlons, qlats, qu, qv, transform=proj, color="black",
                  scale=REF_SPEED * 25, width=0.003, headwidth=4, headlength=4, zorder=5)
    ax.quiverkey(Q, X=0.90, Y=-0.06, U=REF_SPEED, label=f"{REF_SPEED:.0f} m/s",
                 labelpos="E", fontproperties={"size": 8})

    ax.set_title(piece_titles[ip], fontsize=10, loc="left", pad=3)

# Save to step directory (not data/figs/)
png_path = STEP_DIR / "fig8_replica.png"
pdf_path = STEP_DIR / "fig8_replica.pdf"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"✓ Saved: {png_path}  ({png_path.stat().st_size/1e6:.1f} MB)")
print(f"✓ Saved: {pdf_path}  ({pdf_path.stat().st_size/1e6:.1f} MB)")
plt.close(fig)

print("\n=== Step 10 complete — Fig 8 replica ===")
print(f"  8 levels, INLIN=0, 30yr clim")
print(f"  Pieces: lower(1000–850), middle(700–400), upper(300–200)")
