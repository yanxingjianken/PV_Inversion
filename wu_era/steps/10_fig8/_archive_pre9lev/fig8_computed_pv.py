# %% [markdown]
# Step 10c — Self-Consistent PV Advection: Computed PV vs ERA5 PV
#
# **Diagnostic**: Does the ~30-50% overestimate in piecewise PV advection
# come from PV source mismatch (ERA5-downloaded clim PV vs locally-computed PV)?
#
# **Key test**: Recompute PV advection using locally-computed PV for BOTH
# event AND climatology (via identical `ertel_pv()` function), keeping the
# SAME Pass D induced winds (U_ind, V_ind from ψ′ inversion). Then compare:
#   - Σ pieces (new, self-consistent PV) vs Direct (new, self-consistent PV)
#   - Σ pieces (old, ERA5 clim PV)    vs Direct (old, locally-computed PV)
#
# If the overestimate vanishes with self-consistent PV → PV source mismatch.
# If it persists → Fortran non-dim scaling / ψ→wind conversion error.
#
# **Figure**: 3×2 comparison mimicking `fig8_direct_vs_piecewise.png`.
#
# %% [markdown]
# ## 1. Load Data & Compute Self-Consistent PV
#
# %%
import numpy as np, xarray as xr, matplotlib.pyplot as plt
import matplotlib.colors as mcolors, matplotlib.ticker as mticker
import cartopy.crs as ccrs, cartopy.feature as cfeature
from pathlib import Path
from scipy.ndimage import gaussian_filter
import yaml

import sys; from pathlib import Path as _Path
_sys_path_root = str(_Path(__file__).resolve().parent.parent.parent)
if _sys_path_root not in sys.path:
    sys.path.insert(0, _sys_path_root)
import config

STEP_DIR = _Path(__file__).resolve().parent
OUT_DIR = _Path(config.WU_OUT_DIR)
CLIM_DIR = _Path(config.CLIM_DIR)

# ─── Physical constants (from config) ───
G    = config.G
R_E  = config.R_E
OM   = config.OMEGA
P0   = config.P0
KAP  = config.KAP
NW   = config.NW
PLEVS   = config.PLEVS       # [1000, 850, 700, 500, 400, 300, 250, 200] hPa
PLEV_PA = config.PLEVS_PA    # Pa
NX, NY = config.NX, config.NY
K250 = 6  # 250 hPa index

# ─── Load clim & event data on regional grid ───
ds_mean  = xr.open_dataset(CLIM_DIR / "mean_clim.nc")
ds_event = xr.open_dataset(CLIM_DIR / "event.nc")
lats = ds_mean.latitude.values.astype(float)
lons = ds_mean.longitude.values.astype(float)
LON2D, LAT2D = np.meshgrid(lons, lats)

# ─── Ertel PV computation (identical to step 08 parse_and_pv.py) ───
def ertel_pv(t3d, u3d, v3d, plev_Pa_arr, lat_arr):
    """Compute Ertel PV in SI units: K·m²·kg⁻¹·s⁻¹.

    Uses hydrostatic approximation: PV ≈ −g·(ζ+f)·∂θ/∂p
    with centred finite differences on pressure levels.
    """
    theta = t3d * (P0 / plev_Pa_arr[:, None, None]) ** KAP
    dth = np.gradient(theta, plev_Pa_arr, axis=0)
    dla = np.deg2rad(np.gradient(lat_arr))[None, :, None]
    dlo = np.deg2rad(np.gradient(lons))[None, None, :]
    cl  = np.cos(np.deg2rad(lat_arr))[None, :, None]
    zeta = (np.gradient(v3d, axis=2) / (R_E * cl * dlo)
          - np.gradient(u3d, axis=1) / (R_E * dla))
    f = 2 * OM * np.sin(np.deg2rad(lat_arr))[None, :, None]
    return -G * (zeta + f) * dth  # SI — NO ×1e6


# ─── Compute PV from scratch for BOTH event and clim ───
print("Computing PV_event from event T,U,V ...")
PV_event_new = ertel_pv(
    ds_event["t"].values, ds_event["u"].values, ds_event["v"].values,
    PLEV_PA, lats
)
print("Computing PV_clim from clim T,U,V ...")
PV_clim_new = ertel_pv(
    ds_mean["t"].values, ds_mean["u"].values, ds_mean["v"].values,
    PLEV_PA, lats
)
ds_mean.close(); ds_event.close()

# Self-consistent PV anomaly at 250 hPa (SI)
Q_anom_new = PV_event_new[K250] - PV_clim_new[K250]

# Display in PVU for readability
print(f"PV_event @250hPa: [{PV_event_new[K250].min()*1e6:.1f}, {PV_event_new[K250].max()*1e6:.1f}] PVU")
print(f"PV_clim  @250hPa: [{PV_clim_new[K250].min()*1e6:.1f}, {PV_clim_new[K250].max()*1e6:.1f}] PVU")
print(f"Q_anom   @250hPa: [{Q_anom_new.min()*1e6:.1f}, {Q_anom_new.max()*1e6:.1f}] PVU")

# %% [markdown]
# ## 2. Load Piecewise Induced Winds (unchanged — from Pass D ψ′)
#
# %%
ds_pv = xr.open_dataset(OUT_DIR / "pv_advection.nc")

# ─── Original (ERA5-based) PV anomaly for comparison ───
Q_anom_old = ds_pv.Q_anom_250.values.copy()        # SI (ERA5 clim − computed event)
Q_clim_old = ds_pv.Q_clim_250.values.copy()         # SI (ERA5 downloaded)

# ─── Induced winds at 250 hPa (same for both old and new) ───
U_ind = ds_pv.U_induced_250.values.copy()           # (3, 51, 87) m/s
V_ind = ds_pv.V_induced_250.values.copy()           # (3, 51, 87) m/s
PVadv_old = ds_pv.PVadv.values.copy()               # (3, 51, 87) SI s⁻¹ (ERA5 PV based)
ds_pv.close()

NPIECES = 3

# ─── Compare ERA5 clim PV vs locally-computed clim PV ───
print(f"\nERA5 clim PV vs local clim PV @250hPa:")
ratio_clim = Q_clim_old / PV_clim_new[K250]
valid_clim = np.isfinite(Q_clim_old) & np.isfinite(PV_clim_new[K250]) & (PV_clim_new[K250] != 0)
print(f"  Ratio ERA5/local: median={np.median(ratio_clim[valid_clim]):.4f}, "
      f"[{np.nanmin(ratio_clim):.4f}, {np.nanmax(ratio_clim):.4f}]")
corr_clim = np.corrcoef(Q_clim_old[valid_clim].ravel(), PV_clim_new[K250][valid_clim].ravel())[0, 1]
print(f"  Spatial correlation: {corr_clim:.4f}")

print(f"\nERA5-based Q_anom vs self-consistent Q_anom @250hPa:")
valid_q = np.isfinite(Q_anom_old) & np.isfinite(Q_anom_new)
corr_q = np.corrcoef(Q_anom_old[valid_q].ravel(), Q_anom_new[valid_q].ravel())[0, 1]
print(f"  Spatial correlation: {corr_q:.4f}")

# %% [markdown]
# ## 3. Compute PV Advection with Self-Consistent PV
#
# %%
# ─── Compute PV gradients on the sphere (same method as step 09) ───
dlat_rad = np.deg2rad(np.mean(np.abs(np.diff(lats))))
dlon_rad = np.deg2rad(np.mean(np.abs(np.diff(lons))))
coslat = np.cos(np.deg2rad(lats))[:, None]

# Self-consistent PV anomaly gradients
dq_dx_new = np.gradient(Q_anom_new, axis=1) / (R_E * coslat * dlon_rad)
dq_dy_new = np.gradient(Q_anom_new, axis=0) / (R_E * dlat_rad)

# Piecewise PV advection with self-consistent PV (SI s⁻¹) — UNSMOOTHED
PVadv_new_raw = np.zeros((NPIECES, NY, NX), dtype=np.float32)
for ip in range(NPIECES):
    PVadv_new_raw[ip] = -(U_ind[ip] * dq_dx_new + V_ind[ip] * dq_dy_new)

# Also recompute OLD PVadv from scratch (to verify step 09 stored values)
dq_dx_old = np.gradient(Q_anom_old, axis=1) / (R_E * coslat * dlon_rad)
dq_dy_old = np.gradient(Q_anom_old, axis=0) / (R_E * dlat_rad)
PVadv_old_recomp = np.zeros((NPIECES, NY, NX), dtype=np.float32)
for ip in range(NPIECES):
    PVadv_old_recomp[ip] = -(U_ind[ip] * dq_dx_old + V_ind[ip] * dq_dy_old)

print(f"\nPVadv_old stored vs recomputed:")
for ip in range(NPIECES):
    rms_s = np.sqrt(np.nanmean(PVadv_old[ip]**2))
    rms_r = np.sqrt(np.nanmean(PVadv_old_recomp[ip]**2))
    print(f"  Piece {ip+1}: stored rms={rms_s:.4e}  recomputed rms={rms_r:.4e}  ratio={rms_s/max(rms_r,1e-30):.6f}")

# Mask halos on all raw arrays
for ip in range(NPIECES):
    for arr in [PVadv_new_raw, PVadv_old, PVadv_old_recomp]:
        arr[ip, [0, -1], :] = np.nan
        arr[ip, :, [0, -1]] = np.nan

# Create SMOOTHED versions for display only (not used in diagnostics)
PVadv_new_smooth = PVadv_new_raw.copy()
PVadv_old_smooth = PVadv_old.copy()
PVadv_new_smooth[2] = gaussian_filter(np.nan_to_num(PVadv_new_smooth[2]), sigma=1.5)
PVadv_old_smooth[2] = gaussian_filter(np.nan_to_num(PVadv_old_smooth[2]), sigma=1.5)

# Smooth induced winds for display
U_ind_disp = U_ind.copy()
V_ind_disp = V_ind.copy()
U_ind_disp[2] = gaussian_filter(np.nan_to_num(U_ind_disp[2]), sigma=1.5)
V_ind_disp[2] = gaussian_filter(np.nan_to_num(V_ind_disp[2]), sigma=1.5)

# ─── Compute direct PV advection (observed wind anomaly × self-consistent PV gradient) ───
ds_mean2 = xr.open_dataset(CLIM_DIR / "mean_clim.nc")
ds_event2 = xr.open_dataset(CLIM_DIR / "event.nc")
u_anom_250 = ds_event2["u"].values[K250] - ds_mean2["u"].values[K250]
v_anom_250 = ds_event2["v"].values[K250] - ds_mean2["v"].values[K250]
ds_mean2.close(); ds_event2.close()

PVadv_direct_new = -(u_anom_250 * dq_dx_new + v_anom_250 * dq_dy_new)  # SI s⁻¹
PVadv_direct_new[[0, -1], :] = np.nan; PVadv_direct_new[:, [0, -1]] = np.nan

# ─── Sum of pieces (UNSMOOTHED for diagnostics) ───
PVadv_sum_new = PVadv_new_raw[0] + PVadv_new_raw[1] + PVadv_new_raw[2]
PVadv_sum_old = PVadv_old[0] + PVadv_old[1] + PVadv_old[2]

# ─── Also compute direct with ERA5 PV for fair comparison ───
# Old direct (from fig8_direct_decomp — uses locally-computed PV already)
# We recompute here with the same ertel_pv for consistency
dq_dx_old = np.gradient(Q_anom_old, axis=1) / (R_E * coslat * dlon_rad)
dq_dy_old = np.gradient(Q_anom_old, axis=0) / (R_E * dlat_rad)
PVadv_direct_old = -(u_anom_250 * dq_dx_old + v_anom_250 * dq_dy_old)
PVadv_direct_old[[0, -1], :] = np.nan; PVadv_direct_old[:, [0, -1]] = np.nan

# %% [markdown]
# ## 4. Scale to Display Units & Compute Statistics
#
# %%
PVADV_DAY_SCALE = 1e6 * 86400.0  # SI s⁻¹ → PVU/day
PV_SCALE = 1e6                    # SI → PVU

# Scale everything to PVU/day for display
def to_pvu_day(field):
    return field * PVADV_DAY_SCALE

# ─── New (self-consistent PV) — SMOOTHED versions for display ───
pvadv_sum_new_disp  = to_pvu_day(np.sum(PVadv_new_smooth, axis=0))
pvadv_direct_new_disp = to_pvu_day(PVadv_direct_new)
pvadv_diff_new_disp = pvadv_sum_new_disp - pvadv_direct_new_disp
pvadv_pieces_new_disp = to_pvu_day(PVadv_new_smooth)

# ─── Old (ERA5 PV) — SMOOTHED versions for display ───
pvadv_sum_old_disp  = to_pvu_day(np.sum(PVadv_old_smooth, axis=0))
pvadv_direct_old_disp = to_pvu_day(PVadv_direct_old)
pvadv_diff_old_disp = pvadv_sum_old_disp - pvadv_direct_old_disp
pvadv_pieces_old_disp = to_pvu_day(PVadv_old_smooth)

# ─── PV anomaly in PVU ───
q_anom_new_pvu = Q_anom_new * PV_SCALE
q_anom_old_pvu = Q_anom_old * PV_SCALE

# ─── Print key diagnostics ───
print("\n" + "=" * 70)
print("DIAGNOSTIC: Σ pieces vs Direct PV Advection @250 hPa")
print("=" * 70)

# RMS ratio (how much larger is Σ pieces vs direct?)
rms_sum_new = np.sqrt(np.nanmean(PVadv_sum_new**2))
rms_dir_new = np.sqrt(np.nanmean(PVadv_direct_new**2))
rms_sum_old = np.sqrt(np.nanmean(PVadv_sum_old**2))
rms_dir_old = np.sqrt(np.nanmean(PVadv_direct_old**2))

ratio_new = rms_sum_new / max(rms_dir_new, 1e-30)
ratio_old = rms_sum_old / max(rms_dir_old, 1e-30)

print(f"\n  Self-consistent PV (NEW):")
print(f"    RMS Σ pieces:  {rms_sum_new:.4e} SI  =  {rms_sum_new*PVADV_DAY_SCALE:.1f} PVU/day")
print(f"    RMS Direct:    {rms_dir_new:.4e} SI  =  {rms_dir_new*PVADV_DAY_SCALE:.1f} PVU/day")
print(f"    RMS ratio Σ/Direct: {ratio_new:.3f}  ({'OVERESTIMATE' if ratio_new > 1.05 else 'CLOSE'})")

print(f"\n  ERA5 PV (OLD):")
print(f"    RMS Σ pieces:  {rms_sum_old:.4e} SI  =  {rms_sum_old*PVADV_DAY_SCALE:.1f} PVU/day")
print(f"    RMS Direct:    {rms_dir_old:.4e} SI  =  {rms_dir_old*PVADV_DAY_SCALE:.1f} PVU/day")
print(f"    RMS ratio Σ/Direct: {ratio_old:.3f}  ({'OVERESTIMATE' if ratio_old > 1.05 else 'CLOSE'})")

# Correlation (use valid mask for advection fields)
valid_adv_new = np.isfinite(PVadv_sum_new) & np.isfinite(PVadv_direct_new)
valid_adv_old = np.isfinite(PVadv_sum_old) & np.isfinite(PVadv_direct_old)
corr_new = np.corrcoef(PVadv_sum_new[valid_adv_new].ravel(), PVadv_direct_new[valid_adv_new].ravel())[0,1]
corr_old = np.corrcoef(PVadv_sum_old[valid_adv_old].ravel(), PVadv_direct_old[valid_adv_old].ravel())[0,1]
print(f"\n  Spatial correlation Σ vs Direct:")
print(f"    NEW (self-consistent PV): {corr_new:.4f}")
print(f"    OLD (ERA5 PV):            {corr_old:.4f}")

# Per-piece RMS comparison (UNSMOOTHED)
print(f"\n  Per-piece RMS [PVU/day] (UNSMOOTHED for diagnostics):")
print(f"    {'Piece':>10s}  {'NEW':>12s}  {'OLD':>12s}  {'Ratio N/O':>10s}")
for ip in range(NPIECES):
    rms_n = np.sqrt(np.nanmean(PVadv_new_raw[ip]**2)) * PVADV_DAY_SCALE
    rms_o = np.sqrt(np.nanmean(PVadv_old[ip]**2)) * PVADV_DAY_SCALE
    print(f"    {'PLM'[ip]+str(ip+1):>10s}  {rms_n:12.1f}  {rms_o:12.1f}  {rms_n/max(rms_o,1e-30):10.3f}")

# Direct PV advection comparison
print(f"\n  Direct PV advection (observed winds):")
print(f"    NEW: [{np.nanmin(pvadv_direct_new_disp):.1f}, {np.nanmax(pvadv_direct_new_disp):.1f}] PVU/day")
print(f"    OLD: [{np.nanmin(pvadv_direct_old_disp):.1f}, {np.nanmax(pvadv_direct_old_disp):.1f}] PVU/day")

# KEY CONCLUSION
print(f"\n  >>> CONCLUSION: RMS ratio changed from {ratio_old:.2f} → {ratio_new:.2f}")
if abs(ratio_new - 1.0) < 0.1:
    print("  >>> PV source mismatch WAS the dominant cause of overestimate.")
    print("  >>> Fix: update step 08 to compute PV_clim locally instead of using ERA5 PV.")
elif ratio_new > 1.15:
    print("  >>> Overestimate PERSISTS with self-consistent PV.")
    print("  >>> Cause is likely Fortran non-dim scaling or ψ→wind conversion.")
    print("  >>> Check: PSI_SCALE, induced wind computation, Pass D convergence.")
else:
    print("  >>> Partial improvement — PV source mismatch contributes but is not sole cause.")

# %% [markdown]
# ## 5. Plot — 3×2 Self-Consistent PV Comparison
#
# %%
# ─── Auto-scale ───
nice = np.array([0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100, 150, 200])

def _auto_vmax(field):
    v = float(np.nanpercentile(np.abs(field), 98))
    if v < 0.5:
        v = 1.0
    return float(nice[np.argmin(np.abs(nice - v))])

vmax_direct_new = _auto_vmax(pvadv_direct_new_disp)
vmax_sum_new    = _auto_vmax(pvadv_sum_new_disp)
vmax_diff_new   = _auto_vmax(pvadv_diff_new_disp)
vmax_diff_old   = _auto_vmax(pvadv_diff_old_disp)
shared_bot_vmax = max(vmax_sum_new, 50.0)

# ─── q_anom contour levels (shared) ───
p95_q = float(np.nanpercentile(np.abs(q_anom_new_pvu), 95))
nice_ci = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
ci = float(nice_ci[np.argmin(np.abs(nice_ci - p95_q / 6))])
cmax = float(np.ceil(p95_q / ci) * ci)
POS_LEVS = np.arange(ci, cmax + ci / 2, ci)
NEG_LEVS = -POS_LEVS[::-1]

# ─── Wind display cap ───
WIND_CAP = 40.0
for ip in range(NPIECES):
    speed = np.sqrt(U_ind_disp[ip]**2 + V_ind_disp[ip]**2)
    sf = np.where(speed > WIND_CAP, WIND_CAP / np.maximum(speed, 1e-9), 1.0)
    U_ind_disp[ip] *= sf
    V_ind_disp[ip] *= sf

# ─── Plot ───
proj = ccrs.PlateCarree()
QUIVER_SKIP = 4
REF_SPEED = 20.0

with open(_Path(_sys_path_root) / "wu_config.yaml") as _f:
    _yaml_cfg = yaml.safe_load(_f)
_pieces = _yaml_cfg["pieces"]

top_titles = [
    "(a) Direct PV advection\n    (self-consistent PV, observed winds)",
    "(b) Σ pieces PV advection\n    (self-consistent PV, Pass D induced winds)",
    "(c) Σ pieces − Direct\n    (NEW: self-consistent PV residual)",
]
bot_titles = [
    f"(d) Lower piece: {_pieces['lower']['hpa']} hPa",
    f"(e) Middle piece: {_pieces['middle']['hpa']} hPa",
    f"(f) Upper piece: {_pieces['upper']['hpa']} hPa",
]

fig, axes = plt.subplots(2, 3, figsize=(23, 13),
                         subplot_kw={"projection": proj})
fig.suptitle(
    "250-hPa PV Advection: Self-Consistent PV (locally computed) vs Pass D Induced Winds\n"
    f"2025-01-08 00Z  |  8 levels  |  30yr Clim  |  IBC=0  |  INLIN=1\n"
    f"Σ/Direct RMS ratio = {ratio_new:.2f}  |  spatial corr = {corr_new:.3f}",
    fontsize=12, fontweight="bold")

# ─── Shared map config ───
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
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}


def _add_q_contours(ax, q_field, solid=True):
    """Add PV anomaly contours: solid positive, dashed negative."""
    q_safe = np.where(np.isnan(q_field), np.nanmean(q_field), q_field)
    if len(POS_LEVS) > 0 and np.nanmax(q_safe) >= POS_LEVS[0]:
        cs_p = ax.contour(LON2D, LAT2D, q_safe, levels=POS_LEVS,
                          colors="black", linewidths=0.7,
                          linestyles="solid" if solid else "dashed",
                          transform=proj)
        ax.clabel(cs_p, inline=True, fontsize=5, fmt="%.1f")
    if len(NEG_LEVS) > 0 and np.nanmin(q_safe) <= NEG_LEVS[-1]:
        cs_n = ax.contour(LON2D, LAT2D, q_safe, levels=NEG_LEVS,
                          colors="black", linewidths=0.7,
                          linestyles="dashed" if solid else "dashed",
                          transform=proj)
        ax.clabel(cs_n, inline=True, fontsize=5, fmt="%.1f")


def _add_filled(ax, field, vmax_p, label):
    """Add filled PV advection field."""
    cbar_levs = np.linspace(-vmax_p, vmax_p, 17)
    norm = mcolors.BoundaryNorm(cbar_levs, plt.cm.RdBu_r.N)
    cf = ax.contourf(LON2D, LAT2D, field, levels=cbar_levs,
                     cmap="RdBu_r", norm=norm, transform=proj, extend="both")
    plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.01,
                 label=label, fraction=0.03)
    return cf


def _add_ind_winds(ax, u, v):
    """Add induced wind quivers."""
    sk = QUIVER_SKIP
    qlons, qlats = LON2D[::sk, ::sk], LAT2D[::sk, ::sk]
    qu, qv = u[::sk, ::sk], v[::sk, ::sk]
    speed = np.sqrt(qu**2 + qv**2)
    qu = np.where(speed < 1.0, np.nan, qu)
    qv = np.where(speed < 1.0, np.nan, qv)
    Q = ax.quiver(qlons, qlats, qu, qv, transform=proj, color="black",
                  scale=REF_SPEED * 25, width=0.003, headwidth=4, headlength=4, zorder=5)
    ax.quiverkey(Q, X=0.88, Y=-0.06, U=REF_SPEED, label=f"{REF_SPEED:.0f} m/s",
                 labelpos="E", fontproperties={"size": 7})


# ═══════════════ Top row: Direct, Σ pieces (NEW), Diff NEW vs OLD ═══════════════
# Panel (a): Direct PV advection (self-consistent PV)
ax = axes[0, 0]
_add_filled(ax, pvadv_direct_new_disp, vmax_direct_new, "PVU day⁻¹")
_add_q_contours(ax, q_anom_new_pvu, solid=True)
ax.set_title(top_titles[0], fontsize=9, loc="left", pad=3, fontweight="bold")

# Panel (b): Σ pieces PV advection (self-consistent PV)
ax = axes[0, 1]
_add_filled(ax, pvadv_sum_new_disp, vmax_sum_new, "PVU day⁻¹")
_add_q_contours(ax, q_anom_new_pvu, solid=True)
ax.set_title(top_titles[1], fontsize=9, loc="left", pad=3, fontweight="bold")

# Panel (c): Σ − Direct (NEW self-consistent residual)
ax = axes[0, 2]
_add_filled(ax, pvadv_diff_new_disp, vmax_diff_new,
            f"PVU day⁻¹\n(ratio={ratio_new:.2f})")
_add_q_contours(ax, q_anom_new_pvu, solid=True)
ax.set_title(top_titles[2], fontsize=9, loc="left", pad=3, fontweight="bold")

# ═══════════════ Bottom row: Per-piece PV advection (self-consistent PV) ═══════════════
piece_names = ["lower", "middle", "upper"]
for j, pname in enumerate(piece_names):
    ax = axes[1, j]
    _add_filled(ax, pvadv_pieces_new_disp[j], shared_bot_vmax, "PVU day⁻¹")
    _add_q_contours(ax, q_anom_new_pvu, solid=True)
    _add_ind_winds(ax, U_ind_disp[j], V_ind_disp[j])
    ax.set_title(bot_titles[j], fontsize=9, loc="left", pad=3, fontweight="bold")

plt.tight_layout(rect=[0, 0.02, 1, 0.95])

# ─── Add text box with diagnostics ───
diag_text = (
    f"Σ/Direct RMS ratio: {ratio_new:.2f} (was {ratio_old:.2f} with ERA5 PV)\n"
    f"Spatial correlation Σ vs Direct: {corr_new:.3f} (was {corr_old:.3f})\n"
    f"ERA5 clim PV / local clim PV ratio: median={np.median(ratio_clim[valid_clim]):.3f}\n"
    f"PV anomaly corr (ERA5 vs local): {corr_q:.3f}"
)
fig.text(0.02, 0.01, diag_text, fontsize=8, fontfamily="monospace",
         va="bottom", ha="left",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))

# ─── Save ───
png_path = STEP_DIR / "fig8_computed_pv.png"
pdf_path = STEP_DIR / "fig8_computed_pv.pdf"
fig.savefig(png_path, dpi=250, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"\n✓ Saved: {png_path}  ({png_path.stat().st_size/1e6:.1f} MB)")
print(f"✓ Saved: {pdf_path}  ({pdf_path.stat().st_size/1e6:.1f} MB)")
plt.close(fig)

print("\n=== fig8_computed_pv complete ===")
print(f"  Key finding: RMS ratio {ratio_old:.2f} → {ratio_new:.2f}")
print(f"  If ratio → 1.0: PV source mismatch was the cause.")
print(f"  If ratio stays >1.1: Fortran scaling / ψ→wind conversion issue.")
