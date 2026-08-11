# %% [markdown]
# Step 10d — Extended Domain → Cali Crop: Old vs New PV Advection Comparison
#
# **Purpose**: Run the extended-domain (120°E–60°W) PV inversion, then crop
# back to the original Cali window (169.5°W–40.5°W) and compare:
#   1. Σ pieces vs Direct PV advection RMS ratio (was 1.32 on old domain)
#   2. Boundary artifacts — do they disappear when CA is interior?
#   3. Remote PV influence — does far-field PV change the answer?
#
# %% [markdown]
# ## 1. Load Extended-Domain Data
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
CLIM_DIR = _Path(config.CLIM_DIR)

G, R_E, OM, P0, KAP = config.G, config.R_E, config.OMEGA, config.P0, config.KAP
PLEV_PA = config.PLEVS_PA
NW, NX_FULL, NY = config.NW, config.NX, config.NY
K250 = 6

# Load full extended-domain data
ds_pv = xr.open_dataset(OUT_DIR / "pv_advection.nc")
lats_full = ds_pv.lat.values.astype(float)
lons_full = ds_pv.lon.values.astype(float)
LON2D_full, LAT2D_full = np.meshgrid(lons_full, lats_full)

Q_anom_full = ds_pv.Q_anom_250.values.copy()
U_ind_full = ds_pv.U_induced_250.values.copy()
V_ind_full = ds_pv.V_induced_250.values.copy()
PVadv_full = ds_pv.PVadv.values.copy()
ds_pv.close()

ds_psi = xr.open_dataset(OUT_DIR / "piecewise_psi.nc")
PSI_full = ds_psi.psi.values.copy()
U_ind_3d = ds_psi.U_induced.values.copy()
V_ind_3d = ds_psi.V_induced.values.copy()
ds_psi.close()

NPIECES = 3
print(f"Extended domain: NY={NY}, NX={NX_FULL}")

# %% [markdown]
# ## 2. Crop to Original Cali Window (169.5°W–40.5°W, 10.5°N–85.5°N)
#
# %%
# Old domain bounds in -180/180 convention
# Old "Cali" sub-window, expressed in the 0-360 convention used everywhere
# downstream. 190.5°E = 169.5°W, 319.5°E = 40.5°W. This sub-window lies
# entirely east of 180°, so it does NOT cross the antimeridian.
CALI_LON_W, CALI_LON_E = 190.5, 319.5
CALI_LAT_S, CALI_LAT_N = 10.5, 85.5

# Extended-domain lons are 0-360 monotonic (120.0…319.5); select the Cali window.
cali_lon_mask = (lons_full >= CALI_LON_W) & (lons_full <= CALI_LON_E)
cali_lat_mask = (lats_full >= CALI_LAT_S) & (lats_full <= CALI_LAT_N)

lats_cali = lats_full[cali_lat_mask]
lons_cali = lons_full[cali_lon_mask]
LON2D_cali, LAT2D_cali = np.meshgrid(lons_cali, lats_cali)
NY_CALI, NX_CALI = len(lats_cali), len(lons_cali)
print(f"Cali window: {lats_cali[0]:.1f}°N–{lats_cali[-1]:.1f}°N, "
      f"{lons_cali[0]:.1f}°–{lons_cali[-1]:.1f}°, {NY_CALI}×{NX_CALI}")

# Crop all fields
def crop_2d(arr):
    return arr[cali_lat_mask, :][:, cali_lon_mask]

def crop_3d_piece(arr):
    return arr[:, cali_lat_mask, :][:, :, cali_lon_mask]

Q_anom_cali = crop_2d(Q_anom_full)
U_ind_cali = crop_3d_piece(U_ind_full)
V_ind_cali = crop_3d_piece(V_ind_full)
PVadv_cali = crop_3d_piece(PVadv_full)

# Also crop the 3D fields for full-level analysis
# (psi at all 8 levels, induced winds at all levels)

# %% [markdown]
# ## 3. Compute Direct PV Advection on Cropped Cali Domain
#
# %%
# Load clim & event for Cali crop (from the full domain files)
ds_mean = xr.open_dataset(CLIM_DIR / "mean_clim.nc")
ds_event = xr.open_dataset(CLIM_DIR / "event.nc")

# Event and clim data are on the full extended domain; crop to Cali
t_ev_full = ds_event["t"].values
u_ev_full = ds_event["u"].values
v_ev_full = ds_event["v"].values
t_cl_full = ds_mean["t"].values
u_cl_full = ds_mean["u"].values
v_cl_full = ds_mean["v"].values
ds_mean.close(); ds_event.close()

# Compute Ertel PV (same function as step 08)
def ertel_pv(t3d, u3d, v3d, plev_Pa_arr, lat_arr, lon_arr):
    theta = t3d * (P0 / plev_Pa_arr[:, None, None]) ** KAP
    dth = np.gradient(theta, plev_Pa_arr, axis=0)
    dla = np.deg2rad(np.gradient(lat_arr))[None, :, None]
    dlo = np.deg2rad(np.gradient(lon_arr))[None, None, :]
    cl  = np.cos(np.deg2rad(lat_arr))[None, :, None]
    zeta = (np.gradient(v3d, axis=2) / (R_E * cl * dlo)
          - np.gradient(u3d, axis=1) / (R_E * dla))
    f = 2 * OM * np.sin(np.deg2rad(lat_arr))[None, :, None]
    return -G * (zeta + f) * dth

# Compute PV on full domain, then crop
PV_ev_full = ertel_pv(t_ev_full, u_ev_full, v_ev_full, PLEV_PA, lats_full, lons_full)
PV_cl_full = ertel_pv(t_cl_full, u_cl_full, v_cl_full, PLEV_PA, lats_full, lons_full)

# Crop to Cali
PV_ev_cali = crop_2d(PV_ev_full[K250])
PV_cl_cali = crop_2d(PV_cl_full[K250])
Q_anom_computed_cali = PV_ev_cali - PV_cl_cali

# Wind anomaly on Cali crop
u_ev_cali = u_ev_full[K250][cali_lat_mask, :][:, cali_lon_mask]
v_ev_cali = v_ev_full[K250][cali_lat_mask, :][:, cali_lon_mask]
u_cl_cali = u_cl_full[K250][cali_lat_mask, :][:, cali_lon_mask]
v_cl_cali = v_cl_full[K250][cali_lat_mask, :][:, cali_lon_mask]
u_anom_cali = u_ev_cali - u_cl_cali
v_anom_cali = v_ev_cali - v_cl_cali

# Compute direct PV advection on Cali domain
dlat_rad = np.deg2rad(np.mean(np.abs(np.diff(lats_cali))))
dlon_rad = np.deg2rad(np.mean(np.abs(np.diff(lons_cali))))
coslat_cali = np.cos(np.deg2rad(lats_cali))[:, None]

dq_dx_cali = np.gradient(Q_anom_computed_cali, axis=1) / (R_E * coslat_cali * dlon_rad)
dq_dy_cali = np.gradient(Q_anom_computed_cali, axis=0) / (R_E * dlat_rad)
PVadv_direct_cali = -(u_anom_cali * dq_dx_cali + v_anom_cali * dq_dy_cali)
PVadv_direct_cali[[0, -1], :] = np.nan; PVadv_direct_cali[:, [0, -1]] = np.nan

# Mask halos on piecewise advection
for ip in range(NPIECES):
    PVadv_cali[ip, [0, -1], :] = np.nan
    PVadv_cali[ip, :, [0, -1]] = np.nan

# Sum of pieces
PVadv_sum_cali = PVadv_cali[0] + PVadv_cali[1] + PVadv_cali[2]

# %% [markdown]
# ## 4. Compute Diagnostics & Compare with Old Domain Results
#
# %%
PVADV_DAY = 1e6 * 86400.0

rms_sum_new = np.sqrt(np.nanmean(PVadv_sum_cali**2))
rms_dir_new = np.sqrt(np.nanmean(PVadv_direct_cali**2))
ratio_new = rms_sum_new / max(rms_dir_new, 1e-30)

valid_new = np.isfinite(PVadv_sum_cali) & np.isfinite(PVadv_direct_cali)
corr_new = np.corrcoef(PVadv_sum_cali[valid_new].ravel(),
                        PVadv_direct_cali[valid_new].ravel())[0, 1]

print("=" * 70)
print("EXTENDED DOMAIN → CALI CROP: Σ pieces vs Direct PV Advection @250 hPa")
print("=" * 70)
print(f"\n  RMS Σ pieces:  {rms_sum_new*PVADV_DAY:.1f} PVU/day")
print(f"  RMS Direct:    {rms_dir_new*PVADV_DAY:.1f} PVU/day")
print(f"  RMS ratio Σ/Direct: {ratio_new:.3f}")
print(f"  Spatial correlation: {corr_new:.4f}")

# Per-piece RMS
print(f"\n  Per-piece RMS [PVU/day] on Cali crop:")
for ip in range(NPIECES):
    rms_p = np.sqrt(np.nanmean(PVadv_cali[ip]**2)) * PVADV_DAY
    print(f"    Piece {ip+1}: {rms_p:.1f}")

# Compare with OLD domain results
# OLD: RMS ratio = 1.32, corr = 0.957
print(f"\n  === COMPARISON WITH OLD (87×51 Cali-only domain) ===")
print(f"  OLD: RMS ratio = 1.32, corr = 0.957")
print(f"  NEW: RMS ratio = {ratio_new:.3f}, corr = {corr_new:.4f}")
if ratio_new < 1.15:
    print(f"  >>> IMPROVEMENT! RMS ratio dropped from 1.32 → {ratio_new:.2f}")
    print(f"  >>> Far-field PV inclusion reduces the overestimate.")
else:
    print(f"  >>> NO significant change. Overestimate is not from missing far-field PV.")
    print(f"  >>> Root cause is likely Fortran scaling or method-inherent.")

# %% [markdown]
# ## 5. Plot — 3×2: Extended vs Cali Crop Comparison
#
# %%
# Smooth for display
PVadv_cali_smooth = PVadv_cali.copy()
PVadv_cali_smooth[2] = gaussian_filter(np.nan_to_num(PVadv_cali_smooth[2]), sigma=1.5)
PVadv_sum_cali_smooth = PVadv_cali_smooth[0] + PVadv_cali_smooth[1] + PVadv_cali_smooth[2]
PVadv_diff_cali = PVadv_sum_cali_smooth - PVadv_direct_cali

U_ind_cali_disp = U_ind_cali.copy()
V_ind_cali_disp = V_ind_cali.copy()
U_ind_cali_disp[2] = gaussian_filter(np.nan_to_num(U_ind_cali_disp[2]), sigma=1.5)
V_ind_cali_disp[2] = gaussian_filter(np.nan_to_num(V_ind_cali_disp[2]), sigma=1.5)

# Wind cap
WIND_CAP = 40.0
for ip in range(NPIECES):
    speed = np.sqrt(U_ind_cali_disp[ip]**2 + V_ind_cali_disp[ip]**2)
    sf = np.where(speed > WIND_CAP, WIND_CAP / np.maximum(speed, 1e-9), 1.0)
    U_ind_cali_disp[ip] *= sf
    V_ind_cali_disp[ip] *= sf

# Scale to PVU/day
def pvu_day(f):
    return f * PVADV_DAY

PV_SCALE = 1e6
q_anom_cali_pvu = Q_anom_computed_cali * PV_SCALE

# Auto-scale
nice = np.array([0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100, 150, 200])
def _auto_vmax(field):
    v = float(np.nanpercentile(np.abs(field), 98))
    if v < 0.5: v = 1.0
    return float(nice[np.argmin(np.abs(nice - v))])

vmax_direct = _auto_vmax(pvu_day(PVadv_direct_cali))
vmax_sum    = _auto_vmax(pvu_day(PVadv_sum_cali_smooth))
vmax_diff   = _auto_vmax(pvu_day(PVadv_diff_cali))
shared_bot  = max(vmax_sum, 50.0)

# Contour levels
p95_q = float(np.nanpercentile(np.abs(q_anom_cali_pvu), 95))
nice_ci = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
ci = float(nice_ci[np.argmin(np.abs(nice_ci - p95_q / 6))])
cmax = float(np.ceil(p95_q / ci) * ci)
POS_LEVS = np.arange(ci, cmax + ci/2, ci)
NEG_LEVS = -POS_LEVS[::-1]

# ─── Plot ───
proj = ccrs.PlateCarree()
QUIVER_SKIP = 4
REF_SPEED = 20.0

top_titles = [
    "(a) Direct PV advection\n    (observed winds, Cali crop)",
    "(b) Σ pieces PV advection\n    (extended domain → Cali crop)",
    f"(c) Σ − Direct  (ratio={ratio_new:.2f})",
]
bot_titles = [
    "(d) Lower piece (1000–850 hPa)",
    "(e) Middle piece (700–400 hPa)",
    "(f) Upper piece (300–200 hPa)",
]

fig, axes = plt.subplots(2, 3, figsize=(23, 13),
                         subplot_kw={"projection": proj})
fig.suptitle(
    f"250-hPa PV Advection: Extended Domain (120°E–60°W) → Cali Crop\n"
    f"2025-01-08 00Z  |  8 levels  |  INLIN=1  |  IBC=0\n"
    f"Σ/Direct RMS ratio = {ratio_new:.2f}  (was 1.32 on Cali-only domain)  |  "
    f"corr = {corr_new:.3f}",
    fontsize=12, fontweight="bold")

for ax_row in axes:
    for ax in ax_row:
        ax.set_extent([-170, -40, 10, 85], crs=proj)
        ax.add_feature(cfeature.COASTLINE, lw=0.5, edgecolor="gray")
        ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="lightgray")
        ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.2, edgecolor="lightgray")
        gl = ax.gridlines(draw_labels=True, lw=0.3, color="lightgray", alpha=0.7, ls="--")
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(range(-170, -30, 20))
        gl.ylocator = mticker.FixedLocator(range(20, 90, 20))
        gl.xlabel_style = {"size": 7}; gl.ylabel_style = {"size": 7}

def _add_q_contours(ax, q_field, solid=True):
    q_safe = np.where(np.isnan(q_field), np.nanmean(q_field), q_field)
    if len(POS_LEVS) > 0 and np.nanmax(q_safe) >= POS_LEVS[0]:
        cs_p = ax.contour(LON2D_cali, LAT2D_cali, q_safe, levels=POS_LEVS,
                          colors="black", linewidths=0.7,
                          linestyles="solid" if solid else "dashed", transform=proj)
        ax.clabel(cs_p, inline=True, fontsize=5, fmt="%.1f")
    if len(NEG_LEVS) > 0 and np.nanmin(q_safe) <= NEG_LEVS[-1]:
        cs_n = ax.contour(LON2D_cali, LAT2D_cali, q_safe, levels=NEG_LEVS,
                          colors="black", linewidths=0.7,
                          linestyles="dashed" if solid else "dashed", transform=proj)
        ax.clabel(cs_n, inline=True, fontsize=5, fmt="%.1f")

def _add_filled(ax, field, vmax_p, label):
    cbar_levs = np.linspace(-vmax_p, vmax_p, 17)
    norm = mcolors.BoundaryNorm(cbar_levs, plt.cm.RdBu_r.N)
    cf = ax.contourf(LON2D_cali, LAT2D_cali, field, levels=cbar_levs,
                     cmap="RdBu_r", norm=norm, transform=proj, extend="both")
    plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.01, label=label, fraction=0.03)

def _add_winds(ax, u, v):
    sk = QUIVER_SKIP
    qlons, qlats = LON2D_cali[::sk, ::sk], LAT2D_cali[::sk, ::sk]
    qu, qv = u[::sk, ::sk], v[::sk, ::sk]
    speed = np.sqrt(qu**2 + qv**2)
    qu = np.where(speed < 1.0, np.nan, qu)
    qv = np.where(speed < 1.0, np.nan, qv)
    Q = ax.quiver(qlons, qlats, qu, qv, transform=proj, color="black",
                  scale=REF_SPEED*25, width=0.003, headwidth=4, headlength=4, zorder=5)
    ax.quiverkey(Q, X=0.88, Y=-0.06, U=REF_SPEED, label=f"{REF_SPEED:.0f} m/s",
                 labelpos="E", fontproperties={"size": 7})

# Top row
ax = axes[0, 0]
_add_filled(ax, pvu_day(PVadv_direct_cali), vmax_direct, "PVU day⁻¹")
_add_q_contours(ax, q_anom_cali_pvu, solid=True)
ax.set_title(top_titles[0], fontsize=9, loc="left", pad=3, fontweight="bold")

ax = axes[0, 1]
_add_filled(ax, pvu_day(PVadv_sum_cali_smooth), vmax_sum, "PVU day⁻¹")
_add_q_contours(ax, q_anom_cali_pvu, solid=True)
ax.set_title(top_titles[1], fontsize=9, loc="left", pad=3, fontweight="bold")

ax = axes[0, 2]
_add_filled(ax, pvu_day(PVadv_diff_cali), vmax_diff, f"PVU day⁻¹\n(ratio={ratio_new:.2f})")
_add_q_contours(ax, q_anom_cali_pvu, solid=True)
ax.set_title(top_titles[2], fontsize=9, loc="left", pad=3, fontweight="bold")

# Bottom row
for j in range(NPIECES):
    ax = axes[1, j]
    _add_filled(ax, pvu_day(PVadv_cali_smooth[j]), shared_bot, "PVU day⁻¹")
    _add_q_contours(ax, q_anom_cali_pvu, solid=True)
    _add_winds(ax, U_ind_cali_disp[j], V_ind_cali_disp[j])
    ax.set_title(bot_titles[j], fontsize=9, loc="left", pad=3, fontweight="bold")

plt.tight_layout(rect=[0, 0.02, 1, 0.94])

# Diagnostic text box
diag_text = (
    f"EXTENDED DOMAIN (120°E–60°W) → Cali crop\n"
    f"Σ/Direct RMS ratio: {ratio_new:.2f}  |  OLD (Cali-only): 1.32\n"
    f"Spatial correlation: {corr_new:.3f}  |  OLD: 0.957\n"
    f"Per-piece RMS [PVU/day]: P1={np.sqrt(np.nanmean(PVadv_cali[0]**2))*PVADV_DAY:.1f}  "
    f"P2={np.sqrt(np.nanmean(PVadv_cali[1]**2))*PVADV_DAY:.1f}  "
    f"P3={np.sqrt(np.nanmean(PVadv_cali[2]**2))*PVADV_DAY:.1f}"
)
fig.text(0.02, 0.01, diag_text, fontsize=7.5, fontfamily="monospace",
         va="bottom", ha="left",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))

png_path = STEP_DIR / "fig8_extended_cali_compare.png"
pdf_path = STEP_DIR / "fig8_extended_cali_compare.pdf"
fig.savefig(png_path, dpi=250, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"\n✓ Saved: {png_path}  ({png_path.stat().st_size/1e6:.1f} MB)")
print(f"✓ Saved: {pdf_path}  ({pdf_path.stat().st_size/1e6:.1f} MB)")
plt.close(fig)

# %% [markdown]
# ## 6. Boundary Artifact Check — ψ at 250 hPa near CA
#
# %%
# Compare ψ structure near the old domain boundary
# In the Cali-only domain, CA was at the western wall (~170°W).
# In the extended domain, CA is in the interior.
# Check if piecewise ψ has artifacts near 170°W in the new domain.

psi_cali = PSI_full[:, K250][:, cali_lat_mask, :][:, :, cali_lon_mask]
psi_sum_cali = np.sum(psi_cali, axis=0)  # (51_cali, 87_cali)

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12),
                           subplot_kw={"projection": ccrs.PlateCarree()})
fig2.suptitle("ψ′ at 250 hPa: Extended Domain → Cali Crop\n"
              "Check for boundary artifacts near old domain walls",
              fontsize=12, fontweight="bold")

plot_data = [
    (psi_cali[0], "Piece 1 (lower) ψ′"),
    (psi_cali[1], "Piece 2 (middle) ψ′"),
    (psi_cali[2], "Piece 3 (upper) ψ′"),
    (psi_sum_cali, "Σ pieces ψ′"),
]

for idx, (ax, (data, title)) in enumerate(zip(axes2.flat, plot_data)):
    ax.set_extent([-170, -40, 10, 85], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, lw=0.5, edgecolor="gray")
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.2, edgecolor="lightgray")
    vm = float(np.nanpercentile(np.abs(data), 99))
    cf = ax.pcolormesh(LON2D_cali, LAT2D_cali, data, cmap="RdBu_r",
                       vmin=-vm, vmax=vm, transform=ccrs.PlateCarree())
    plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.06, fraction=0.04)
    ax.set_title(title, fontsize=9)

plt.tight_layout()
psi_path = STEP_DIR / "fig8_extended_psi_boundary_check.png"
fig2.savefig(psi_path, dpi=200, bbox_inches="tight")
print(f"✓ Saved: {psi_path}")
plt.close(fig2)

print("\n=== fig8_extended_cali_compare complete ===")
print(f"  KEY RESULT: RMS ratio = {ratio_new:.2f} (OLD was 1.32)")
if ratio_new < 1.15:
    print(f"  >>> Far-field PV INCLUSION improves closure (reduces overestimate)")
else:
    print(f"  >>> Overestimate PERSISTS — not from missing far-field PV")
