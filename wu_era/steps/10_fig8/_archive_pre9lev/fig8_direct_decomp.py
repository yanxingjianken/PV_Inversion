# %% [markdown]
# Step 10b — 2×3 Panel: Direct vs Piecewise PV Advection at 250 hPa
#
# **Top row**: Direct ERA5 PV advection computed from event-day (t,u,v,z)
#   - Left:   −u′·∂q′/∂x  (zonal PV advection component)
#   - Middle: −v′·∂q′/∂y  (meridional PV advection component)
#   - Right:  −(u′·∂q′/∂x + v′·∂q′/∂y)  (total direct PV advection)
#   - Black solid contours: 250 hPa PV anomaly (ERA5)
#
# **Bottom row**: Piecewise PV advection by Pass D induced winds
#   - Left:   Lower piece  (1000–850 hPa)
#   - Middle: Middle piece (700–400 hPa)
#   - Right:  Upper piece  (300–200 hPa)
#   - Black solid contours: 250 hPa PV anomaly
#   - Black dashed contours: mean PV anomaly over each piece's levels
#
# %% [markdown]
# ## 1. Load & Compute All Data
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
WU_DIR = _Path(config.WU_IN_DIR)

# ——— Load clim & event data on regional grid ———
ds_mean = xr.open_dataset(CLIM_DIR / "mean_clim.nc")
ds_event = xr.open_dataset(CLIM_DIR / "event.nc")
lats = ds_mean.latitude.values.astype(float)
lons = ds_mean.longitude.values.astype(float)
LON2D, LAT2D = np.meshgrid(lons, lats)
NY, NX = len(lats), len(lons)

PLEV_PA = config.PLEVS_PA  # [100000, 85000, ..., 20000] Pa, 8 levels
PLEVS   = config.PLEVS     # [1000, 850, 700, 500, 400, 300, 250, 200] hPa
NW = config.NW
K250 = 6  # 250 hPa index (0-based: 1000,850,700,500,400,300,250,200)

R_E = 6.371e6
G  = 9.81
P0 = 1e5
RD = 287.0
CP = 1004.0
KAP = RD / CP
OMEGA = 7.292e-5
lat_rad = np.deg2rad(lats)
f_cor = 2 * OMEGA * np.sin(lat_rad)
cos_lat = np.cos(lat_rad)
dlat = float(np.deg2rad(float(np.mean(np.abs(np.diff(lats))))))
dlon = float(np.deg2rad(float(np.mean(np.abs(np.diff(lons))))))

# ——— Compute Ertel PV from (t, u, v, z) ———
def compute_pv(t, u, v, z):
    """Compute Ertel PV [PVU] from t[K], u,v[m/s], z[m] on (lev,lat,lon)."""
    theta = t * (P0 / PLEV_PA[:, None, None]) ** KAP
    # dθ/dp: centred finite differences along level axis
    dthdp = np.zeros_like(theta)
    dp_vec = PLEV_PA.astype(float)
    for k in range(1, NW - 1):
        dthdp[k] = (theta[k + 1] - theta[k - 1]) / (dp_vec[k + 1] - dp_vec[k - 1])
    dthdp[0] = (theta[1] - theta[0]) / (dp_vec[1] - dp_vec[0])
    dthdp[-1] = (theta[-1] - theta[-2]) / (dp_vec[-1] - dp_vec[-2])
    
    # Relative vorticity ζ = ∂v/∂x − ∂u/∂y
    dv_di, dv_dj = np.gradient(v, axis=(1, 2))
    du_di, du_dj = np.gradient(u, axis=(1, 2))
    dy = float(dlat * R_E)
    dx_j = float(dlon) * R_E * cos_lat
    zeta = dv_dj / dx_j[None, :, None] - du_di / dy
    
    # Ertel PV in SI (K·m²·kg⁻¹·s⁻¹)
    pv = -G * (zeta + f_cor[None, :, None]) * dthdp
    return pv

pv_mean  = compute_pv(ds_mean["t"].values, ds_mean["u"].values,
                       ds_mean["v"].values, ds_mean["z"].values / G)
pv_event = compute_pv(ds_event["t"].values, ds_event["u"].values,
                       ds_event["v"].values, ds_event["z"].values / G)

# ——— Anomalies at 250 hPa ———
u_anom_250 = ds_event["u"].values[K250] - ds_mean["u"].values[K250]
v_anom_250 = ds_event["v"].values[K250] - ds_mean["v"].values[K250]
q_anom_250 = pv_event[K250] - pv_mean[K250]  # PVU

# ——— Direct PV advection at 250 hPa (SI: K·m²·kg⁻¹·s⁻²) ———
dqdx_250 = np.gradient(q_anom_250, float(dlon), axis=1) / (R_E * cos_lat[:, None])
dqdy_250 = np.gradient(q_anom_250, float(dlat), axis=0) / R_E
pvadv_u = -u_anom_250 * dqdx_250       # SI s⁻¹ — NO *86400
pvadv_v = -v_anom_250 * dqdy_250
pvadv_direct = pvadv_u + pvadv_v

# ——— Scale to display-friendly units (PVU, PVU/day) for plotting ———
PV_SCALE = 1e6                # SI → PVU
PVADV_DAY_SCALE = 1e6 * 86400.0  # SI s⁻¹ → PVU/day
pvadv_u *= PVADV_DAY_SCALE
pvadv_v *= PVADV_DAY_SCALE
pvadv_direct *= PVADV_DAY_SCALE
q_anom_250 *= PV_SCALE

# ——— Load piecewise PV advection from Pass D (SI) ———
ds_pv = xr.open_dataset(OUT_DIR / "pv_advection.nc")
PVADV_DAY_SCALE = 1e6 * 86400.0  # SI s⁻¹ → PVU/day
PV_SCALE = 1e6                    # SI → PVU
PVadv_pieces_SI = ds_pv.PVadv.values.copy()     # (3, 51, 87) SI s⁻¹
PVadv_pieces = PVadv_pieces_SI * PVADV_DAY_SCALE  # scale to PVU/day for display
U_ind  = ds_pv.U_induced_250.values.copy()
V_ind  = ds_pv.V_induced_250.values.copy()

# ——— Smooth piece 3 ———
PVadv_pieces[2] = gaussian_filter(np.nan_to_num(PVadv_pieces[2]), sigma=1.5)
U_ind[2] = gaussian_filter(np.nan_to_num(U_ind[2]), sigma=1.5)
V_ind[2] = gaussian_filter(np.nan_to_num(V_ind[2]), sigma=1.5)

# ——— Sum of 3 pieces + difference vs direct total ———
pvadv_sum  = PVadv_pieces[0] + PVadv_pieces[1] + PVadv_pieces[2]
pvadv_diff = pvadv_sum - pvadv_direct

# ——— Per-panel auto-scale ———
nice = np.array([0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100, 150, 200])

def _auto_vmax(field):
    v = float(np.nanpercentile(np.abs(field), 98))
    if v < 0.5:
        v = 1.0
    return float(nice[np.argmin(np.abs(nice - v))])

vmax_u = _auto_vmax(pvadv_u)
vmax_v = _auto_vmax(pvadv_v)
vmax_direct = _auto_vmax(pvadv_direct)
vmax_sum    = _auto_vmax(pvadv_sum)
vmax_diff   = _auto_vmax(pvadv_diff)
vmax_pieces = [_auto_vmax(PVadv_pieces[ip]) for ip in range(3)]

# ——— Shared q_anom contour levels ———
p95_q = float(np.nanpercentile(np.abs(q_anom_250), 95))
nice_ci = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
ci = float(nice_ci[np.argmin(np.abs(nice_ci - p95_q / 6))])
cmax = float(np.ceil(p95_q / ci) * ci)
POS_LEVS = np.arange(ci, cmax + ci / 2, ci)
NEG_LEVS = -POS_LEVS[::-1]

# ——— Per-piece mean PV anomaly (for dashed contours in bottom row) ———
# Piece definitions from YAML
with open(_Path(_sys_path_root) / "wu_config.yaml") as _f:
    _yaml_cfg = yaml.safe_load(_f)
_pieces = _yaml_cfg["pieces"]
piece_levels = {
    "lower":  np.array(_pieces["lower"]["levels"])   - 1,  # 0-based
    "middle": np.array(_pieces["middle"]["levels"])  - 1,
    "upper":  np.array(_pieces["upper"]["levels"])   - 1,
}
q_anom_full = pv_event - pv_mean  # (8, 51, 87) PVU
piece_mean_q = {}
for pname, kidx in piece_levels.items():
    piece_mean_q[pname] = np.nanmean(q_anom_full[kidx], axis=0)  # (51, 87)

# Scale per-piece mean PV anomaly for dashed contours
# (PVadv_pieces already scaled to PVU/day when loaded)
for pname in piece_mean_q:
    piece_mean_q[pname] *= PV_SCALE

# ——— Wind cap for induced winds (bottom row) ———
WIND_CAP = 40.0
for ip in range(3):
    speed = np.sqrt(U_ind[ip]**2 + V_ind[ip]**2)
    sf = np.where(speed > WIND_CAP, WIND_CAP / np.maximum(speed, 1e-9), 1.0)
    U_ind[ip] *= sf
    V_ind[ip] *= sf

# ——— Load Pass A Fortran PV (from meanq / event_q.out) for top-row overlay ———
def _read_wu_ascii_nv(fp):
    """Read Wu ASCII file, return numerical values (skip header)."""
    data = []
    with open(fp) as f:
        for line in f:
            for tok in line.split():
                data.append(float(tok))
    return np.array(data[8:])  # skip 8-value header

# Pass A PV: stored as [THB(51,87), THT(51,87), Q(6 interior levels, 51, 87)]
# Skip header(8) + 2 theta blocks, then reshape remaining 6 levels
NW_PV = config.NW_PV
block = NY * NX
_raw = _read_wu_ascii_nv(WU_DIR / "meanq")
q_mean_wu = _raw[2*block:].reshape(NW_PV, NY, NX)          # (6, 51, 87) Wu units
_raw_e = _read_wu_ascii_nv(WU_DIR / "event_q.out")
q_event_wu = _raw_e[2*block:].reshape(NW_PV, NY, NX)
q_anom_wu = q_event_wu - q_mean_wu  # (6, 51, 87) in Wu PV units
# 250 hPa is the last interior level (index 5)
q_anom_wu_250 = q_anom_wu[5]  # Wu PV anomaly at 250 hPa

# ——— Mask boundary halos ———
def _mask_boundary(arr):
    arr[0, :] = arr[-1, :] = np.nan
    arr[:, 0] = arr[:, -1] = np.nan
for arr in [pvadv_u, pvadv_v, pvadv_direct]:
    _mask_boundary(arr)
for ip in range(3):
    _mask_boundary(PVadv_pieces[ip])
    _mask_boundary(U_ind[ip])
    _mask_boundary(V_ind[ip])

# NaN-safe q contour
q_contour = np.where(np.isnan(q_anom_250), np.nanmean(q_anom_250), q_anom_250)

# %% [markdown]
# ## 2. Plot — 2×3 Panel
#
# %%
proj = ccrs.PlateCarree()
QUIVER_SKIP = 4
REF_SPEED = 20.0

# Titles
top_titles = [
    r"(a) −(u′·∂q′/∂x + v′·∂q′/∂y)  (direct ERA5)",
    r"(b) Σ pieces  (lower + middle + upper)",
    r"(c) Σ pieces − direct  (difference)",
]
bot_titles = [
    f"(d) Lower piece: {_pieces['lower']['hpa']} hPa",
    f"(e) Middle piece: {_pieces['middle']['hpa']} hPa",
    f"(f) Upper piece: {_pieces['upper']['hpa']} hPa",
]

fig, axes = plt.subplots(2, 3, figsize=(22, 13),
                         subplot_kw={"projection": proj})
fig.suptitle(
    "250-hPa PV Advection: Direct ERA5 vs Sum-of-Pieces (Pass D)\n"
    f"2025-01-08 00Z  |  8 levels  |  30yr Clim  |  IBC=0",
    fontsize=12, fontweight="bold")

# ——— Shared map config ———
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
    if len(POS_LEVS) > 0 and np.nanmax(q_field) >= POS_LEVS[0]:
        cs_p = ax.contour(LON2D, LAT2D, q_field, levels=POS_LEVS,
                          colors="black", linewidths=0.7,
                          linestyles="solid" if solid else "dashed",
                          transform=proj)
        ax.clabel(cs_p, inline=True, fontsize=5, fmt="%.1f")
    if len(NEG_LEVS) > 0 and np.nanmin(q_field) <= NEG_LEVS[-1]:
        cs_n = ax.contour(LON2D, LAT2D, q_field, levels=NEG_LEVS,
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


# ═══════════════ Top row: Direct total, sum of pieces, difference ═══════════════
top_fields = [pvadv_direct, pvadv_sum, pvadv_diff]
top_vmaxs  = [vmax_direct, vmax_sum, vmax_diff]
top_labels = ["PVU day⁻¹", "PVU day⁻¹", "PVU day⁻¹"]
for j in range(3):
    ax = axes[0, j]
    _add_filled(ax, top_fields[j], top_vmaxs[j], top_labels[j])
    _add_q_contours(ax, q_contour, solid=True)
    ax.set_title(top_titles[j], fontsize=9, loc="left", pad=3,
                 fontweight="bold")

# ═══════════════ Bottom row: Piecewise PV advection ═══════════════
# Use shared vmax from sum-of-pieces for cross-comparison
shared_bot_vmax = max(vmax_sum, 50.0)
piece_names = ["lower", "middle", "upper"]
for j, pname in enumerate(piece_names):
    ax = axes[1, j]
    _add_filled(ax, PVadv_pieces[j], shared_bot_vmax, "PVU day⁻¹")
    _add_q_contours(ax, q_contour, solid=True)
    _add_q_contours(ax, piece_mean_q[pname], solid=False)
    _add_ind_winds(ax, U_ind[j], V_ind[j])
    ax.set_title(bot_titles[j], fontsize=9, loc="left", pad=3,
                 fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.96])

# ——— Save ———
png_path = STEP_DIR / "fig8_direct_vs_piecewise.png"
pdf_path = STEP_DIR / "fig8_direct_vs_piecewise.pdf"
fig.savefig(png_path, dpi=250, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"✓ Saved: {png_path}  ({png_path.stat().st_size/1e6:.1f} MB)")
print(f"✓ Saved: {pdf_path}  ({pdf_path.stat().st_size/1e6:.1f} MB)")
plt.close(fig)

# ——— Quick stats ———
print(f"\nDirect PV adv ranges [PVU/day]:")
print(f"  Total direct:  [{np.nanmin(pvadv_direct):.1f}, {np.nanmax(pvadv_direct):.1f}]")
print(f"  Σ pieces:      [{np.nanmin(pvadv_sum):.1f}, {np.nanmax(pvadv_sum):.1f}]")
print(f"  Σ − direct:    [{np.nanmin(pvadv_diff):.1f}, {np.nanmax(pvadv_diff):.1f}]")
print(f"\nPiecewise PV adv ranges [PVU/day]:")
for j, pname in enumerate(piece_names):
    print(f"  {pname}: [{np.nanmin(PVadv_pieces[j]):.1f}, {np.nanmax(PVadv_pieces[j]):.1f}]")

# Residual: sum of piecewise vs direct
print(f"\nΣ pieces vs Direct total:")
print(f"  RMS residual: {np.sqrt(np.nanmean(pvadv_diff**2)):.1f} PVU/day")
print(f"  Correlation:  {np.corrcoef(pvadv_sum.ravel(), pvadv_direct.ravel())[0,1]:.4f}")
print(f"\n=== fig8_direct_decomp complete ===")
