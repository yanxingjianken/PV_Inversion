# %% [markdown]
# Step 09 — PV Advection by Induced Winds
#
# Compute the 250-hPa PV advection by the piecewise-induced winds:
#
# $$\text{PVadv} = -\left(u' \frac{\partial q}{\partial x} + v' \frac{\partial q}{\partial y}\right) \times 86400 \quad [\text{PVU/day}]$$
#
# where u′, v′ are the induced winds from each PV piece (Step 08) and
# q is the total ERA5 Ertel PV at 250 hPa (NOT the Wu internal Q!).
#
# This is the central quantity plotted in Davis Fig 8: it shows whether
# each piece's induced circulation is advecting PV cyclonically (+)
# or anticyclonically (−) — the mechanism by which PV pieces sustain blocking.
#
# %% [markdown]
# ## 1. Load Data
#
# %%
import numpy as np, xarray as xr, matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from pathlib import Path
from scipy.ndimage import gaussian_filter

import sys; from pathlib import Path as _Path
_sys_path_root = str(_Path(__file__).resolve().parent.parent.parent)
if _sys_path_root not in sys.path: sys.path.insert(0, _sys_path_root)
import config
STEP_DIR = _Path(__file__).resolve().parent
DATA_DIR = _Path(config.DATA_DIR); ERA5_DIR = _Path(config.ERA5_DIR)
CLIM_DIR = _Path(config.CLIM_DIR); WU_DIR = _Path(config.WU_IN_DIR)
OUT_DIR = _Path(config.WU_OUT_DIR); FIG_DIR = _Path(config.FIG_DIR)
BUILD_DIR = _Path(config.DATA_DIR) / "wu_bin"

STEP_DIR = Path.cwd()
OUT_DIR = OUT_DIR

ds = xr.open_dataset(OUT_DIR / "pv_advection.nc")
lats = ds.lat.values; lons = ds.lon.values
LON2D, LAT2D = np.meshgrid(lons, lats)

q_event = ds.Q_event_250.values    # ERA5 PV @250 hPa [PVU]
q_anom  = ds.Q_anom_250.values
U_ind   = ds.U_induced_250.values  # (3, 51, 87) [m/s]
V_ind   = ds.V_induced_250.values

NY, NX = q_event.shape
NPIECES = 3

print(f"Q_event 250: [{np.nanmin(q_event):.2f}, {np.nanmax(q_event):.2f}] PVU")
print(f"Q_anom 250:  [{np.nanmin(q_anom):.2f}, {np.nanmax(q_anom):.2f}] PVU")

# %% [markdown]
# ## 2. Compute PV Gradient
#
# %%
R_EARTH = 2.0e7 / np.pi
DP = R_EARTH * np.radians(1.5)
DL = R_EARTH * np.radians(1.5)
AP = np.cos(np.radians(lats))

# Fill small NaN regions for gradient computation
q_fill = np.where(np.isnan(q_event), np.nanmean(q_event), q_event)

dqdx = np.zeros_like(q_fill)
dqdy = np.zeros_like(q_fill)
dqdy[1:-1, :] = (q_fill[:-2, :] - q_fill[2:, :]) / (2.0 * DP)
for i in range(1, NY-1):
    dqdx[i, 1:-1] = (q_fill[i, 2:] - q_fill[i, :-2]) / (2.0 * DL * AP[i])

print(f"dq/dx: [{np.nanmin(dqdx[1:-1,1:-1]):.2e}, {np.nanmax(dqdx[1:-1,1:-1]):.2e}] PVU/m")
print(f"dq/dy: [{np.nanmin(dqdy[1:-1,1:-1]):.2e}, {np.nanmax(dqdy[1:-1,1:-1]):.2e}] PVU/m")

# %% [markdown]
# ## 3. Compute PV Advection Per Piece
#
# %%
S_PER_DAY = 86400.0
PVadv = np.zeros((NPIECES, NY, NX))

for ip in range(NPIECES):
    PVadv[ip] = -(U_ind[ip] * dqdx + V_ind[ip] * dqdy) * S_PER_DAY
    # Mask 1-cell halo
    PVadv[ip, 0, :] = PVadv[ip, -1, :] = np.nan
    PVadv[ip, :, 0] = PVadv[ip, :, -1] = np.nan

# Apply gentle smoothing to piece 3 (suppress upper-BC noise)
PVadv_smooth = PVadv.copy()
PVadv_smooth[2] = gaussian_filter(np.nan_to_num(PVadv[2]), sigma=1.5)
PVadv_smooth[2, 0, :] = PVadv_smooth[2, -1, :] = np.nan
PVadv_smooth[2, :, 0] = PVadv_smooth[2, :, -1] = np.nan

print("PV advection per piece [PVU/day]:")
for ip in range(NPIECES):
    pv = PVadv[ip, 1:-1, 1:-1]
    print(f"  Piece {ip+1}: min/p1/p50/p99/max = "
          f"{np.nanmin(pv):.1f}/{np.nanpercentile(pv,1):.1f}/"
          f"{np.nanpercentile(pv,50):.1f}/{np.nanpercentile(pv,99):.1f}/"
          f"{np.nanmax(pv):.1f}")

# %% [markdown]
# ## 4. Visualize: PV Gradient + PV Advection (All 3 Pieces)
#
# %%
proj = ccrs.LambertConformal(central_longitude=-105, central_latitude=50)
pc   = ccrs.PlateCarree()

# Row 1: dq/dx, dq/dy. Row 2: PVadv for 3 pieces
fig = plt.figure(figsize=(20, 14))

for col, (data, title, cmap, vmx) in enumerate([
    (dqdx, "∂q/∂x [PVU/m]", "RdBu_r", np.nanpercentile(np.abs(dqdx), 98)),
    (dqdy, "∂q/∂y [PVU/m]", "RdBu_r", np.nanpercentile(np.abs(dqdy), 98)),
]):
    ax = fig.add_subplot(2, 4, col+1, projection=proj)
    ax.set_extent([-175, -35, 5, 88], crs=pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor="0.5")
    cf = ax.pcolormesh(LON2D, LAT2D, data, cmap=cmap, transform=pc,
                       vmin=-vmx, vmax=vmx)
    plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02)
    ax.set_title(title, fontsize=10, fontweight="bold")

piece_titles = ["(a) Lower (1000–925 hPa)", "(b) Middle (850–700 hPa)",
                "(c) Upper (600–250 hPa) [smoothed]"]
for ip in range(3):
    ax = fig.add_subplot(2, 4, ip+5, projection=proj)
    ax.set_extent([-175, -35, 5, 88], crs=pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4")
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.2, edgecolor="0.6")
    vmx = np.nanpercentile(np.abs(PVadv_smooth[ip]), 98)
    cf = ax.pcolormesh(LON2D, LAT2D, PVadv_smooth[ip], cmap="RdBu_r",
                       transform=pc, vmin=-vmx, vmax=vmx)
    plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02, label="PVU/day")
    ax.set_title(f"{piece_titles[ip]}\nPVadv [{np.nanmin(PVadv_smooth[ip]):.0f}, "
                 f"{np.nanmax(PVadv_smooth[ip]):.0f}] PVU/day", fontsize=10)

# Hide unused subplot
fig.delaxes(fig.add_subplot(2, 4, 4))

plt.suptitle("250-hPa PV Gradient & PV Advection by Piecewise-Induced Winds\n"
             "2025-01-08 00Z CA Blocking Event", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(STEP_DIR/"pv_advection_all_pieces.png", dpi=150, bbox_inches="tight")
print("✓ Saved: pv_advection_all_pieces.png")
plt.show()

# %% [markdown]
# ## 5. Save to NetCDF
#
# %%
ds_out = ds.load().copy()  # Eager load before closing
ds_out["PVadv"] = (["piece","lat","lon"], PVadv.astype(np.float32))
ds_out["PVadv_smooth"] = (["piece","lat","lon"], PVadv_smooth.astype(np.float32))
ds.close()  # Release file handle before overwriting
ds_out.to_netcdf(OUT_DIR / "pv_advection.nc")
print(f"✓ Updated: {OUT_DIR / 'pv_advection.nc'}")
print("\n→ Step 10: Fig 8 replica — the grand finale")

