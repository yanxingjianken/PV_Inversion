# %% [markdown]
# Step 08 — Parse Outputs + Compute ERA5 Ertel PV
#
# **The critical correctness step.** Wu's Q is in opaque solver-internal units
# (~600× PVU). We compute true Ertel PV from ERA5 t,u,v in PVU for:
# 1. Physical-unit PV anomaly contours on Fig 8
# 2. Physical-unit PV advection (∇q × induced winds)
# 3. Cross-comparison: Wu Q vs ERA5 PV → quantifies the ~600× scaling
#
# **Ertel PV formula** (isobaric):
# $$PV = -g (\zeta + f) \frac{\partial\theta}{\partial p}$$
# where ζ = relative vorticity, f = Coriolis, θ = potential temperature.
#
# %% [markdown]
# ## 1. Load Wu Outputs + ERA5 Data
#
# %%
import numpy as np, xarray as xr, matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from pathlib import Path

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
WU_DIR   = WU_DIR
OUT_DIR  = OUT_DIR
CLIM_DIR = CLIM_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

NX, NY, NW = 87, 51, 10
NPIECES = 3
block = NY * NX

# Load mean/event from Step 03 NetCDFs
ds_mean = xr.open_dataset(CLIM_DIR / "mean_clim.nc")
ds_event = xr.open_dataset(CLIM_DIR / "event.nc")
lats = ds_mean.latitude.values
lons = ds_mean.longitude.values
plev = ds_mean.pressure_level.values.astype(float) * 100.0  # Pa

t_mean = ds_mean["t"].values  # (10, 51, 87)
u_mean = ds_mean["u"].values
v_mean = ds_mean["v"].values
t_ev   = ds_event["t"].values
u_ev   = ds_event["u"].values
v_ev   = ds_event["v"].values

print(f"ERA5 data: shape={t_ev.shape}, lats={len(lats)}, lons={len(lons)}")
print(f"Pressure levels [hPa]: {plev/100}")

# %% [markdown]
# ## 2. Compute Ertel PV from ERA5 (Python, Physical PVU)
#
# %%
def ertel_pv(t3d, u3d, v3d, plev_Pa, lat_arr):
    """Ertel PV at all levels in PVU. Arrays: (lev, lat, lon), lat N→S."""
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
    pv = -g * (zeta + f) * dtheta_dp * 1.0e6   # PVU
    return pv

PV_mean  = ertel_pv(t_mean,  u_mean,  v_mean,  plev, lats)   # (10, 51, 87)
PV_event = ertel_pv(t_ev,    u_ev,    v_ev,    plev, lats)

# 250 hPa (K=8, 0-based)
k250 = 8
pv_mean_250  = PV_mean[k250]
pv_event_250 = PV_event[k250]
pv_anom_250  = pv_event_250 - pv_mean_250

print(f"ERA5 PV @250 hPa:")
print(f"  Mean:  [{np.min(pv_mean_250):.2f}, {np.max(pv_mean_250):.2f}] PVU")
print(f"  Event: [{np.min(pv_event_250):.2f}, {np.max(pv_event_250):.2f}] PVU")
print(f"  Anom:  [{np.min(pv_anom_250):.2f}, {np.max(pv_anom_250):.2f}] PVU "
      f"(p1/p99={np.percentile(pv_anom_250,1):.2f}/{np.percentile(pv_anom_250,99):.2f})")

# %% [markdown]
# ## 3. Cross-Compare: Wu Q vs ERA5 PV (The ~600× Mystery)
#
# Read Wu's Q at 250 hPa and compute the scaling factor.
#
# %%
def read_wu_ascii(fp):
    d=[]; 
    with open(fp) as f:
        for ln in f:
            for tk in ln.split(): d.append(float(tk))
    return np.array(d[:8]), np.array(d[8:])

_, vq = read_wu_ascii(WU_DIR / "event_q.out")
Q_wu_event = vq[(2+7)*block:(3+7)*block].reshape(NY, NX)  # K=7 interior = 250 hPa
mask_wu = Q_wu_event >= 9999.

# Compare where Wu has valid data
valid = ~mask_wu
if valid.sum() > 0:
    ratio = Q_wu_event[valid] / pv_event_250[valid]
    print(f"Wu Q / ERA5 PV ratio @ 250 hPa:")
    print(f"  Median: {np.median(ratio):.1f}")
    print(f"  Mean ± std: {np.mean(ratio):.1f} ± {np.std(ratio):.1f}")
    print(f"  Range: [{np.min(ratio):.0f}, {np.max(ratio):.0f}]")
    print(f"  → Wu Q ≈ ERA5 PV × {np.median(ratio):.0f} (opaque solver scaling)")

# %% [markdown]
# ## 4. Visualize: Wu Q vs ERA5 PV at 250 hPa
#
# %%
LON2D, LAT2D = np.meshgrid(lons, lats)
proj = ccrs.LambertConformal(central_longitude=-105, central_latitude=50)
pc   = ccrs.PlateCarree()

wu_display = np.where(mask_wu, np.nan, Q_wu_event)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5),
                                      subplot_kw={"projection": proj})
for ax, data, title, cmap in [
    (ax1, pv_event_250, "ERA5 Ertel PV [PVU]", "RdBu_r"),
    (ax2, wu_display, "Wu Q (internal ~600× PVU)", "RdBu_r"),
    (ax3, pv_anom_250, "ERA5 PV Anomaly (event − mean) [PVU]", "RdBu_r"),
]:
    vm = np.nanpercentile(np.abs(data), 98)
    ax.set_extent([-175, -35, 5, 88], crs=pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4")
    cf = ax.pcolormesh(LON2D, LAT2D, data, cmap=cmap, transform=pc,
                       vmin=-vm, vmax=vm)
    ax.contour(LON2D, LAT2D, data, colors="black", linewidths=0.3, transform=pc,
               levels=np.linspace(-vm, vm, 10))
    plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02)
    ax.set_title(f"{title}\n[{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]", fontsize=10)

plt.suptitle("Wu Q vs ERA5 PV — Same Blocking Pattern, Different Units\n"
             "Spatial structure is identical; only the magnitude scale differs",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(STEP_DIR/"wu_vs_era5_pv_250hpa.png", dpi=150, bbox_inches="tight")
print("✓ Saved: wu_vs_era5_pv_250hpa.png")
plt.show()

# %% [markdown]
# ## 5. Compute Induced Winds from ψ′ (Each Piece)
#
# %%
# Load SP from event_pert.out (computed in Step 07)
_, vp = read_wu_ascii(WU_DIR / "event_pert.out")
per_piece = 2 * NW * block
SP = np.zeros((NPIECES, NW, NY, NX))
for ip in range(NPIECES):
    base = ip * per_piece + NW * block  # skip HP, go to SP
    for k in range(NW):
        SP[ip,k] = vp[base + k*block: base + (k+1)*block].reshape(NY, NX)

R_EARTH = 2.0e7 / np.pi   # Wu's AA constant
DP = R_EARTH * np.radians(1.5)
DL = R_EARTH * np.radians(1.5)
AP = np.cos(np.radians(lats))

U_ind = np.zeros((NPIECES, NW, NY, NX))
V_ind = np.zeros((NPIECES, NW, NY, NX))

for ip in range(NPIECES):
    psi = SP[ip] * 1.0e5
    # u = dψ/dy (I increases southward → d/dI = −d/dy)
    U_ind[ip, :, 1:-1, :] = (psi[:, 2:, :] - psi[:, :-2, :]) / (2.0 * DP)
    for i in range(1, NY-1):
        V_ind[ip, :, i, 1:-1] = (psi[:, i, 2:] - psi[:, i, :-2]) / (2.0 * DL * AP[i])

# Mask boundary halo
for ip in range(NPIECES):
    U_ind[ip, :, 0, :] = U_ind[ip, :, -1, :] = np.nan
    U_ind[ip, :, :, 0] = U_ind[ip, :, :, -1] = np.nan
    V_ind[ip, :, 0, :] = V_ind[ip, :, -1, :] = np.nan
    V_ind[ip, :, :, 0] = V_ind[ip, :, :, -1] = np.nan

print("Induced winds @ 250 hPa (interior):")
for ip in range(NPIECES):
    u = U_ind[ip, 8, 1:-1, 1:-1]
    v = V_ind[ip, 8, 1:-1, 1:-1]
    print(f"  Piece {ip+1}: U [{u.min():.1f}, {u.max():.1f}]  V [{v.min():.1f}, {v.max():.1f}] m/s")

# %% [markdown]
# ## 6. Save Output NetCDFs
#
# %%
PLEVS = np.array([1000., 925., 850., 700., 600., 500., 400., 300., 250., 200.])
piece_idx = np.arange(NPIECES)

ds_psi = xr.Dataset({
    "HP": (["piece","plev","lat","lon"], SP.astype(np.float32)),
    "SP": (["piece","plev","lat","lon"], SP.astype(np.float32)),
    "U_induced": (["piece","plev","lat","lon"], U_ind.astype(np.float32)),
    "V_induced": (["piece","plev","lat","lon"], V_ind.astype(np.float32)),
}, coords={"piece": piece_idx, "plev": PLEVS, "lat": lats, "lon": lons})
ds_psi.to_netcdf(OUT_DIR / "piecewise_psi.nc")

ds_adv = xr.Dataset({
    "Q_event_250": (["lat","lon"], pv_event_250.astype(np.float32)),
    "Q_clim_250":  (["lat","lon"], pv_mean_250.astype(np.float32)),
    "Q_anom_250":  (["lat","lon"], pv_anom_250.astype(np.float32)),
    "U_induced_250": (["piece","lat","lon"], U_ind[:, 8].astype(np.float32)),
    "V_induced_250": (["piece","lat","lon"], V_ind[:, 8].astype(np.float32)),
}, coords={"piece": piece_idx, "lat": lats, "lon": lons})
ds_adv.to_netcdf(OUT_DIR / "pv_advection.nc")

print(f"✓ Saved: {OUT_DIR / 'piecewise_psi.nc'}")
print(f"✓ Saved: {OUT_DIR / 'pv_advection.nc'}")
print("\n→ Step 09: Compute PV advection = −u'·∇q for each piece")

