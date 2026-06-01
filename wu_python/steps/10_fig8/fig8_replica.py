#!/usr/bin/env python3
"""Wu Python — Step 10: Fig 8 Replica — Davis (2022) style 3-panel figure.

Final figure: PV advection (filled) + PV anomaly contours + wind vectors per piece.

Generates:
  - fig8_replica.png  — 3-panel cartopy figure
  - fig8_replica.pdf  — PDF version
"""
import numpy as np, xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs, cartopy.feature as cfeature
from pathlib import Path
from scipy.ndimage import gaussian_filter
import sys

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path: sys.path.insert(0, str(_root))
from wu_python.core.grid import NY, NX, lats, lons, LON2D, LAT2D

STEP_DIR = Path(__file__).resolve().parent

# Try multiple NetCDF sources
nc_candidates = [
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "wu_python_out" / "pv_advection.nc",
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "wu_out" / "pv_advection.nc",
]
nc_path = None
for p in nc_candidates:
    if p.exists(): nc_path = p; break
if nc_path is None: raise FileNotFoundError("pv_advection.nc not found")

ds = xr.open_dataset(nc_path)
q_event = ds.Q_event_250.values; q_anom = ds.Q_anom_250.values
ds.close()

# Compute gradients
R_EARTH = 2.0e7 / np.pi; DP = R_EARTH * np.radians(1.5); DL = R_EARTH * np.radians(1.5)
AP = np.cos(np.radians(lats))
q_fill = np.where(np.isnan(q_event), np.nanmean(q_event), q_event)
dqdx = np.zeros_like(q_fill); dqdy = np.zeros_like(q_fill)
dqdy[1:-1, :] = (q_fill[:-2, :] - q_fill[2:, :]) / (2.0 * DP)
for i in range(1, NY-1): dqdx[i, 1:-1] = (q_fill[i, 2:] - q_fill[i, :-2]) / (2.0 * DL * AP[i])

# Load induced winds
import config as root_config
WU_DIR = Path(root_config.WU_IN_DIR)
def read_wu_ascii(fp):
    data = []
    with open(fp) as f:
        for line in f:
            for tok in line.split(): data.append(float(tok))
    return np.array(data[:8]), np.array(data[8:])

NW = 10; block = NY * NX; NPIECES = 3
_, vals = read_wu_ascii(WU_DIR / "event_pert.out")
n_per = 2 * NW * block
PSI = np.stack([vals[ip*n_per + (NW+7)*block:(ip*n_per + (NW+8)*block)].reshape(NY, NX) for ip in range(NPIECES)], axis=0)

# Induced winds
U_ind = np.zeros((NPIECES, NY, NX)); V_ind = np.zeros((NPIECES, NY, NX))
for ip in range(NPIECES):
    for i in range(1, NY-1):
        U_ind[ip, i, 1:-1] = -(PSI[ip, i-1, 1:-1] - PSI[ip, i+1, 1:-1]) / (2.0 * DP)
        V_ind[ip, i, 1:-1] = (PSI[ip, i, 2:] - PSI[ip, i, :-2]) / (2.0 * DL * AP[i])

# Cap extreme winds (piece 3 noise)
for ip in range(NPIECES):
    spd = np.sqrt(U_ind[ip]**2 + V_ind[ip]**2)
    mask = spd > 40.0
    U_ind[ip][mask] *= 40.0 / spd[mask]
    V_ind[ip][mask] *= 40.0 / spd[mask]

# PV advection
S = 86400.0; PVadv = np.zeros((NPIECES, NY, NX))
for ip in range(NPIECES):
    PVadv[ip] = -(U_ind[ip]*dqdx + V_ind[ip]*dqdy) * S
    PVadv[ip, 0, :] = PVadv[ip, -1, :] = PVadv[ip, :, 0] = PVadv[ip, :, -1] = np.nan
PVadv_s = PVadv.copy(); PVadv_s[2] = gaussian_filter(np.nan_to_num(PVadv[2]), sigma=1.5)
PVadv_s[2, 0, :] = PVadv_s[2, -1, :] = PVadv_s[2, :, 0] = PVadv_s[2, :, -1] = np.nan

# ---- Fig 8: 3-panel Davis-style ----
vmax_adv = [5.0, 7.5, 50.0]
p95_q = np.nanpercentile(np.abs(q_anom), 95)
clevs = np.arange(0.5, p95_q + 0.5, 0.5)
clevs_symmetric = sorted(list(set(list(-clevs) + list(clevs))))
print(f"q_anom |p95| = {p95_q:.2f} PVU; contour levels: {clevs}")

proj = ccrs.LambertConformal(central_longitude=-105, central_latitude=50); pc = ccrs.PlateCarree()
fig, axes = plt.subplots(1, 3, figsize=(22, 7), subplot_kw={"projection": proj})
titles = ["(a) Lower (1000–925 hPa)", "(b) Middle (850–700 hPa)", "(c) Upper (600–250 hPa)"]

for ip in range(3):
    ax = axes[ip]
    ax.set_extent([-175, -35, 5, 88], crs=pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.5, edgecolor="0.3")
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.2, edgecolor="0.5")

    # PV advection (filled)
    vmx = vmax_adv[ip]
    cf = ax.pcolormesh(LON2D, LAT2D, PVadv_s[ip], cmap="RdBu_r", transform=pc,
                       vmin=-vmx, vmax=vmx, shading="auto")

    # PV anomaly contours (positive solid, negative dashed)
    q_anom_masked = np.where(np.isnan(q_anom), 0, q_anom)
    ax.contour(LON2D, LAT2D, q_anom_masked, levels=[c for c in clevs_symmetric if c > 0],
               colors="black", linewidths=0.6, transform=pc)
    ax.contour(LON2D, LAT2D, q_anom_masked, levels=[c for c in clevs_symmetric if c < 0],
               colors="black", linewidths=0.6, linestyles="dashed", transform=pc)

    # Wind vectors (subsample every 4th)
    skip = 4
    ax.quiver(LON2D[::skip, ::skip], LAT2D[::skip, ::skip],
              U_ind[ip, ::skip, ::skip], V_ind[ip, ::skip, ::skip],
              transform=pc, scale=200, width=0.003, color="black", alpha=0.7)

    plt.colorbar(cf, ax=ax, shrink=0.75, pad=0.03, label="PVU/day")
    ax.set_title(titles[ip], fontsize=11, fontweight="bold")

plt.suptitle("Wu Python — PV Advection by Piecewise-Induced Winds\n"
             "2025-01-08 00Z CA Blocking  |  Filled: PV adv [PVU/day]  |  "
             "Contours: ΔPV [PVU] (solid +, dashed −)  |  Vectors: induced wind",
             fontsize=12, fontweight="bold")
plt.tight_layout()

out_png = STEP_DIR / "fig8_replica.png"
out_pdf = STEP_DIR / "fig8_replica.pdf"
plt.savefig(out_png, dpi=200, bbox_inches="tight")
plt.savefig(out_pdf, dpi=200, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved: {out_png}  ({out_png.stat().st_size / 1e6:.1f} MB)")
print(f"  ✓ Saved: {out_pdf}  ({out_pdf.stat().st_size / 1e6:.1f} MB)")
print("\n✓ Step 10 complete — wu_python pipeline DONE!")
