# %% [markdown]
# Step 11 — Cross-Section: Meridional Wind Induced by Upper/Middle/Lower PV Anomalies
#
# **Cross-section at Southern California latitude (34.5°N).**
# Plots 3 panels (upper/middle/lower pieces) showing:
# - **Filled**: V_induced (meridional wind, m/s) — log-p y-axis (200→1000 hPa)
# - **Contours**: PV anomaly (PVU) for that piece's pressure levels
#
# Reads: `piecewise_psi.nc` + ERA5 `mean_clim.nc` / `event.nc`
#
# %% [markdown]
# ## 1. Load Data
#
# %%
import numpy as np, xarray as xr, matplotlib.pyplot as plt, yaml
from pathlib import Path
from matplotlib.ticker import ScalarFormatter

import sys
_sys_path_root = str(Path(__file__).resolve().parent.parent.parent)
if _sys_path_root not in sys.path:
    sys.path.insert(0, _sys_path_root)
import config

STEP_DIR = Path(__file__).resolve().parent
OUT_DIR = Path(config.WU_OUT_DIR)
CLIM_DIR = Path(config.CLIM_DIR)
FIG_DIR = Path(config.FIG_DIR)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load YAML for xsection config (lives in wu/)
_YAML_CFG_PATH = Path(_sys_path_root) / "wu_config.yaml"
with open(_YAML_CFG_PATH) as _f:
    _yaml_cfg = yaml.safe_load(_f)
_xcfg = _yaml_cfg["xsection"]
_pieces = _yaml_cfg["pieces"]

TARGET_LAT = _xcfg["lat"]
NPIECES = 3

# ── Load V_induced from piecewise_psi.nc ──
ds_psi = xr.open_dataset(OUT_DIR / "piecewise_psi.nc")
V_ind = ds_psi["V_induced"].values  # (piece, plev, lat, lon)
plevs = ds_psi["plev"].values       # hPa
lats_arr = ds_psi["lat"].values
lons = ds_psi["lon"].values

# ── Find nearest latitude index ──
lat_idx = int(np.argmin(np.abs(lats_arr - TARGET_LAT)))
actual_lat = lats_arr[lat_idx]
print(f"Target lat: {TARGET_LAT}°N → nearest grid point: {actual_lat}°N (index {lat_idx})")

# ── Compute PV anomaly at all levels from clim data ──
ds_mean = xr.open_dataset(CLIM_DIR / "mean_clim.nc")
ds_event = xr.open_dataset(CLIM_DIR / "event.nc")

def ertel_pv_3d(t3d, u3d, v3d, plev_Pa, lat_arr):
    """Ertel PV in PVU. Returns (plev, lat, lon)."""
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
    pv = -g * (zeta + f) * dtheta_dp * 1.0e6
    return pv

plev_Pa = ds_mean.pressure_level.values.astype(float) * 100.0

PV_mean = ertel_pv_3d(
    ds_mean["t"].values, ds_mean["u"].values, ds_mean["v"].values,
    plev_Pa, lats_arr)
PV_event = ertel_pv_3d(
    ds_event["t"].values, ds_event["u"].values, ds_event["v"].values,
    plev_Pa, lats_arr)
PV_anom = PV_event - PV_mean  # (10, 51, 87)

print(f"PV anomaly range @ all levels: [{PV_anom.min():.2f}, {PV_anom.max():.2f}] PVU")

# %% [markdown]
# ## 2. Build Cross-Section Arrays
#
# %%
# Slice at target latitude
V_xsec = V_ind[:, :, lat_idx, :]    # (3, 10, 87) — piece × plev × lon
PV_xsec = PV_anom[:, lat_idx, :]    # (10, 87) — plev × lon

# Piece definitions from YAML (Fortran 1-indexed K → Python 0-indexed k)
piece_names = ["upper", "middle", "lower"]  # display order: upper at top, lower at bottom
piece_labels = {
    "upper": f"Upper ({_pieces['upper']['hpa']} hPa)",
    "middle": f"Middle ({_pieces['middle']['hpa']} hPa)",
    "lower": f"Lower ({_pieces['lower']['hpa']} hPa)",
}
# Map piece name → K indices (1-indexed → 0-indexed)
piece_k = {
    pname: [k - 1 for k in _pieces[pname]["levels"]]
    for pname in piece_names
}

print("Piece level indices (0-indexed):")
for pname in piece_names:
    km = piece_k[pname]
    print(f"  {pname}: K={[k+1 for k in km]} → {[plevs[k] for k in km]} hPa")

# %% [markdown]
# ## 3. Plot — 3-Panel Cross-Section (Meridional Wind + PV Anomaly Contours)
#
# %%
LON2D, PLEV2D = np.meshgrid(lons, plevs)

fig, axes = plt.subplots(1, 3, figsize=(22, 8), sharey=True)

for ip, (ax, pname) in enumerate(zip(axes, piece_names)):
    # Filled: V_induced from this piece
    v_piece = V_xsec[ip]  # (10, 87)
    vm = np.nanpercentile(np.abs(v_piece), 99)
    if not np.isfinite(vm) or vm <= 0:
        vm = 1.0

    cf = ax.pcolormesh(LON2D, PLEV2D, v_piece, cmap="RdBu_r",
                       vmin=-vm, vmax=vm, shading="auto")
    plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02, label="V_induced [m/s]")

    # ── Contours: total PV anomaly in plev×lon space ──
    # Solid = positive PV anomaly, Dashed = negative PV anomaly
    pv_vm = np.nanpercentile(np.abs(PV_xsec), 99)
    if np.isfinite(pv_vm) and pv_vm > 0.01:
        n_levels = 12
        pos_levels = np.linspace(pv_vm / n_levels, pv_vm, n_levels // 2)
        neg_levels = np.linspace(-pv_vm, -pv_vm / n_levels, n_levels // 2)
        ax.contour(LON2D, PLEV2D, PV_xsec, levels=pos_levels,
                   colors="black", linewidths=0.8, linestyles="-")
        ax.contour(LON2D, PLEV2D, PV_xsec, levels=neg_levels,
                   colors="black", linewidths=0.8, linestyles="--")

    # ── Piece boundary markers (dotted horizontal lines) ──
    km = piece_k[pname]
    kmin = min(km)
    kmax = max(km)
    p_top = plevs[kmax]  # top of piece (lower pressure)
    p_bot = plevs[kmin]  # bottom of piece (higher pressure)
    ax.axhline(y=p_top, color="gray", linewidth=1.5, linestyle=":", alpha=0.7)
    ax.axhline(y=p_bot, color="gray", linewidth=1.5, linestyle=":", alpha=0.7)

    # ── Info text ──
    ax.text(0.98, 0.02,
            "PV anom: solid (+) / dashed (−)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="black", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # ── Formatting ──
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_ylim(200, 1000)
    ax.set_xlabel("Longitude [°E]")
    ax.set_title(piece_labels[pname], fontsize=12, fontweight="bold")

    # Clean log ticks
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticks([200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
    ax.set_yticklabels([200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], fontsize=7)

    ax.set_xlim(lons[0], lons[-1])
    ax.grid(True, alpha=0.3, which="both")

# Shared y-label
axes[0].set_ylabel("Pressure [hPa]")

plt.suptitle(
    f"Meridional Wind Induced by Piecewise PV Inversion — Cross-Section at {actual_lat:.1f}°N\n"
    f"CA Blocking Event 2025-01-08 00Z  |  INLIN=1  |  Davis (1991) nonlinear balance",
    fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])

fname_png = FIG_DIR / "xsection_vinduced_3panel.png"
fname_pdf = FIG_DIR / "xsection_vinduced_3panel.pdf"
plt.savefig(fname_png, dpi=200, bbox_inches="tight")
plt.savefig(fname_pdf, bbox_inches="tight")
print(f"✓ Saved: {fname_png}")
print(f"✓ Saved: {fname_pdf}")
plt.show()

# %% [markdown]
# ## 4. Summary Statistics
#
# %%
for ip, pname in enumerate(piece_names):
    vp = V_xsec[ip]
    print(f"\n{pname} piece (@ {actual_lat}°N):")
    print(f"  V_induced range: [{np.nanmin(vp):.2f}, {np.nanmax(vp):.2f}] m/s")
    print(f"  V_induced p95: ±{np.nanpercentile(np.abs(vp), 95):.2f} m/s")
    km = piece_k[pname]
    for k in km:
        pv_max = np.nanmax(np.abs(PV_xsec[k, :]))
        print(f"  {plevs[k]:.0f} hPa PV anom max |Δ|: {pv_max:.2f} PVU")

print("\n✓ Step 11 complete.")
