# %% [markdown]
# Step 11 — Cross-Section: Meridional Wind Induced by surface/lower/upper PV pieces
#
# **Cross-section at Southern California latitude (34.5°N).**
# Plots 3 panels (upper/lower/surface pieces). For each piece, piecewise PV
# inversion inverts ONLY that piece's PV anomaly (zero elsewhere), so each panel
# shows a matched (source PV, induced wind) pair:
# - **Filled**: v' induced by inverting this piece's PV anomaly (m/s), full column
# - **Contours**: this piece's *source* PV anomaly (PVU), confined to its levels
# - log-p y-axis (1000→100 hPa)
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
PV_anom = PV_event - PV_mean  # (nlev, nlat, nlon) — SI units

print(f"PV anomaly range @ all levels: "
      f"[{PV_anom.min()*1e6:.2f}, {PV_anom.max()*1e6:.2f}] PVU  "
      f"(SI: [{PV_anom.min():.2e}, {PV_anom.max():.2e}])")

# %% [markdown]
# ## 2. Build Cross-Section Arrays
#
# %%
# Slice at target latitude
V_xsec = V_ind[:, :, lat_idx, :]        # (3, nlev, nlon) — piece × plev × lon
PV_xsec = PV_anom[:, lat_idx, :] * 1e6  # (nlev, nlon) — plev × lon, in PVU

# Piece definitions from YAML (Fortran 1-indexed K → Python 0-indexed k).
# piecewise_psi.nc stores pieces in the order wu_pass_d.py wrote them = the YAML
# pieces order (surface, lower, upper). Map name → file index explicitly so the
# induced wind and its source overlay stay matched.
# BALP convention: K=1 (index 0, 1000 hPa) = lower-boundary θ; K=NL=9 (index 8,
# 100 hPa) = upper-boundary θ; interior PV is K=2..8 (indices 1..7, 850..200).
nlev = len(plevs)
INTERIOR = set(range(1, nlev - 1))  # indices with genuine inverted PV (850..200)

FILE_PIECE_ORDER = list(_pieces.keys())  # surface, lower, upper (= on-disk piece index)
piece_data_idx = {name: FILE_PIECE_ORDER.index(name) for name in FILE_PIECE_ORDER}

piece_names = list(reversed(FILE_PIECE_ORDER))  # display: upper at left → surface at right
piece_labels = {p: f"{p.capitalize()} ({_pieces[p]['hpa']} hPa)" for p in FILE_PIECE_ORDER}
# Map piece name → K indices (1-indexed → 0-indexed)
piece_k = {p: [k - 1 for k in _pieces[p]["levels"]] for p in FILE_PIECE_ORDER}

print("Piece level indices (0-indexed) and file index:")
for pname in piece_names:
    km = piece_k[pname]
    print(f"  {pname}: file_idx={piece_data_idx[pname]}  "
          f"K={[k+1 for k in km]} → {[plevs[k] for k in km]} hPa")

# %% [markdown]
# ## 3. Plot — 3-Panel Cross-Section (Meridional Wind + PV Anomaly Contours)
#
# %%
LON2D, PLEV2D = np.meshgrid(lons, plevs)

# Common PV contour scaling across all panels (PVU) so intervals are comparable.
pv_vm = np.nanpercentile(np.abs(PV_xsec), 99)
n_levels = 12
pos_levels = np.linspace(pv_vm / n_levels, pv_vm, n_levels // 2)
neg_levels = np.linspace(-pv_vm, -pv_vm / n_levels, n_levels // 2)

fig, axes = plt.subplots(1, 3, figsize=(22, 8), sharey=True)

for ip, (ax, pname) in enumerate(zip(axes, piece_names)):
    # Filled: V_induced from this piece (index into file's piece order, not display order)
    v_piece = V_xsec[piece_data_idx[pname]]  # (nlev, nlon)
    vm = np.nanpercentile(np.abs(v_piece), 99)
    if not np.isfinite(vm) or vm <= 0:
        vm = 1.0

    cf = ax.pcolormesh(LON2D, PLEV2D, v_piece, cmap="RdBu_r",
                       vmin=-vm, vmax=vm, shading="auto")
    plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02, label="V_induced [m/s]")

    # ── Contours: THIS piece's source PV anomaly (PVU), INTERIOR levels only ──
    # Each panel's v' is induced by inverting ONLY this piece's source (PV at its
    # interior levels + any boundary θ). Contour only the genuine interior PV
    # levels (850..200); the boundary-θ levels (1000, 100) carry no Ertel PV that
    # was inverted, so we do NOT draw a (spurious) PV contour there — they are
    # marked by the gray band + the panel label instead.
    km = piece_k[pname]
    km_pv = [k for k in km if k in INTERIOR]
    has_sfc_theta = (0 in km)             # piece includes K=1 lower-boundary θ
    has_top_theta = (nlev - 1 in km)      # piece includes K=NL upper-boundary θ
    if km_pv and np.isfinite(pv_vm) and pv_vm > 0.01:
        PV_piece = np.full_like(PV_xsec, np.nan)
        PV_piece[km_pv, :] = PV_xsec[km_pv, :]
        ax.contour(LON2D, PLEV2D, PV_piece, levels=pos_levels,
                   colors="black", linewidths=0.8, linestyles="-")
        ax.contour(LON2D, PLEV2D, PV_piece, levels=neg_levels,
                   colors="black", linewidths=0.8, linestyles="--")

    # ── Piece band: shade the pressure range this piece spans ──
    # (axhspan stays visible even when a boundary coincides with the y-limits.)
    p_top = plevs[max(km)]  # top of piece (lower pressure)
    p_bot = plevs[min(km)]  # bottom of piece (higher pressure)
    ax.axhspan(min(p_bot, p_top), max(p_bot, p_top), color="gray", alpha=0.12, zorder=0)
    ax.axhline(y=p_top, color="gray", linewidth=1.5, linestyle=":", alpha=0.7)
    ax.axhline(y=p_bot, color="gray", linewidth=1.5, linestyle=":", alpha=0.7)

    # ── Info text: describe this piece's actual source ──
    _src = []
    if km_pv:
        _src.append("interior PV: solid (+) / dashed (−)")
    if has_sfc_theta:
        _src.append(f"{plevs[0]:.0f} hPa boundary θ")
    if has_top_theta:
        _src.append(f"{plevs[-1]:.0f} hPa top θ")
    ax.text(0.98, 0.02, "source — " + "; ".join(_src),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="black", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # ── Formatting: log-p axis, 1000 hPa at bottom (surface), decreasing upward ──
    ax.set_yscale("log")
    ax.set_ylim(1000, 100)  # 1000 at bottom, 100 (top θ) at top
    ax.set_xlabel("Longitude [°E]")
    ax.set_title(piece_labels[pname], fontsize=12, fontweight="bold")

    # Pressure ticks at the model levels, plain (non-scientific) labels
    _yt = [1000, 850, 700, 500, 400, 300, 250, 200, 100]
    ax.set_yticks(_yt)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticklabels(_yt, fontsize=7)
    ax.minorticks_off()

    ax.set_xlim(lons[0], lons[-1])
    ax.grid(True, alpha=0.3, which="both")

# Shared y-label
axes[0].set_ylabel("Pressure [hPa]")

plt.suptitle(
    f"Meridional Wind Induced by Piecewise PV Inversion — Cross-Section at {actual_lat:.1f}°N\n"
    f"CA Blocking Event 2025-01-08 00Z  |  INLIN=1  |  30yr Clim  |  9 Levels (no interp)",
    fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])

fname_png = STEP_DIR / "xsection_vinduced_3panel.png"
fname_pdf = STEP_DIR / "xsection_vinduced_3panel.pdf"
plt.savefig(fname_png, dpi=200, bbox_inches="tight")
plt.savefig(fname_pdf, bbox_inches="tight")
print(f"✓ Saved: {fname_png}")
print(f"✓ Saved: {fname_pdf}")
plt.show()

# %% [markdown]
# ## 4. Summary Statistics
#
# %%
for pname in piece_names:
    vp = V_xsec[piece_data_idx[pname]]
    print(f"\n{pname} piece (@ {actual_lat}°N):")
    print(f"  V_induced range: [{np.nanmin(vp):.2f}, {np.nanmax(vp):.2f}] m/s")
    print(f"  V_induced p95: ±{np.nanpercentile(np.abs(vp), 95):.2f} m/s")
    km = piece_k[pname]
    for k in km:
        pv_max = np.nanmax(np.abs(PV_xsec[k, :]))
        print(f"  {plevs[k]:.0f} hPa PV anom max |Δ|: {pv_max:.2f} PVU")

print("\n✓ Step 11 complete.")
