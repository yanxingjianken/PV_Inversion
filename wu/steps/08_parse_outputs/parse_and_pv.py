# %% [markdown]
# Step 08 — Parse Outputs + Pre-Computed PV + Helmholtz ψ
#
# **SI conventions (2026-06-03):**
#   ψ (streamfunction)  → m²/s       (Wu file × PSI_SCALE)
#   Φ (geopotential)    → m²/s²      (Wu H [m] × G)
#   PV (Ertel)          → K·m²·kg⁻¹·s⁻¹  (ERA5 native SI)
#   PVadv (advection)   → K·m²·kg⁻¹·s⁻²  (SI tendency)
#
# Uses 30-year (1990-2020) monthly PV climatology.
# **No vertical interpolation** — clim PV levels = Wu levels (8 levels).
# Also loads Helmholtz rotational wind for ψ verification.
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
DATA_DIR = Path(config.DATA_DIR); CLIM_DIR = Path(config.CLIM_DIR)
WU_DIR = Path(config.WU_IN_DIR); OUT_DIR = Path(config.WU_OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLIM_SRC = Path("/net/flood/data2/users/x_yan/era/clim")
MON = {1:"jan",2:"feb",3:"mar",4:"apr",5:"may",6:"jun",
       7:"jul",8:"aug",9:"sep",10:"oct",11:"nov",12:"dec"}
EM = MON[config.EVENT_MONTH]

NX, NY, NW = config.NX, config.NY, config.NW
NPIECES, block = 3, NY * NX
DLAT, DLON = config.DLAT, config.DLON
WU_LEVS = config.PLEVS  # [1000,850,700,500,400,300,250,200]

# SI constants from config
G   = config.G
R_E = config.R_E
OM  = config.OMEGA
RD  = config.RD
CP  = config.CP
P0  = config.P0
KAP = config.KAP
PSI_SCALE = config.PSI_SCALE  # 1.0e5: Wu ψ file → m²/s

# %% [markdown]
# ## 1. Load Pre-Computed 30yr PV Climatology (no interp — levels match)
# %%
ds_mean = xr.open_dataset(CLIM_DIR / "mean_clim.nc")
ds_event = xr.open_dataset(CLIM_DIR / "event.nc")
lats = ds_mean.latitude.values
lons = ds_mean.longitude.values

# --- Load clim PV: drop 100 hPa (clim 9 → Wu 8 levels) ---
# ERA5 CDS PV is native SI: K·m²·kg⁻¹·s⁻¹ (~1e-6). Keep in SI.
ds_pv = xr.open_dataset(CLIM_SRC / f"era5_hourly_clim_1990-2020_{EM}_pv.nc")
pv_raw = ds_pv["pv"].sel(day=config.EVENT_DAY, hour=0).squeeze().values
clim_plev_all = ds_pv.pressure_level.values.astype(float)  # [1000,850,700,500,400,300,250,200,100]
clim_lat  = ds_pv.latitude.values.astype(float)
clim_lon  = ds_pv.longitude.values.astype(float)

# Drop 100 hPa → 8 levels matching Wu
keep_8 = clim_plev_all != 100.0
pv_8 = pv_raw[keep_8]
clim_plev_8 = clim_plev_all[keep_8]
assert np.allclose(clim_plev_8, WU_LEVS), f"Clim PV level mismatch: {list(clim_plev_8)} vs {list(WU_LEVS)}"
PV_mean = pv_8  # ERA5 PV is already SI (K·m²·kg⁻¹·s⁻¹); NO *1e6

lat_mask = (clim_lat <= config.LAT_N) & (clim_lat >= config.LAT_S)
_lw, _le = config.LON_W % 360, config.LON_E % 360
_clon = clim_lon % 360
lon_mask = (_clon >= _lw) & (_clon <= _le)
PV_mean = PV_mean[:, lat_mask, :][:, :, lon_mask]
print(f"PV mean shape: {PV_mean.shape} (expecting ({NW},{NY},{NX}))")

# Event PV from t,u,v (no pre-computed event PV) — compute in SI
t_ev, u_ev, v_ev = ds_event["t"].values, ds_event["u"].values, ds_event["v"].values
plev_Pa = WU_LEVS * 100.0

def ertel_pv(t3d, u3d, v3d, plev_Pa_arr, lat_arr):
    """Compute Ertel PV in SI units: K m2 kg-1 s-1."""
    theta = t3d * (P0 / plev_Pa_arr[:, None, None]) ** KAP
    dth = np.gradient(theta, plev_Pa_arr, axis=0)
    dla = np.deg2rad(np.gradient(lat_arr))[None, :, None]
    dlo = np.deg2rad(np.gradient(lons))[None, None, :]
    cl = np.cos(np.deg2rad(lat_arr))[None, :, None]
    zeta = (np.gradient(v3d, axis=2) / (R_E * cl * dlo)
          - np.gradient(u3d, axis=1) / (R_E * dla))
    f = 2 * OM * np.sin(np.deg2rad(lat_arr))[None, :, None]
    return -G * (zeta + f) * dth  # SI — NO *1e6

PV_event = ertel_pv(t_ev, u_ev, v_ev, plev_Pa, lats)

k250 = 6  # 250 hPa is index 6 of 8 [0=1000,1=850,2=700,3=500,4=400,5=300,6=250,7=200]
# Display in practical units: 1e-6 SI = 1 PVU
print(f"PV @250hPa: Mean(30yr)=[{PV_mean[k250].min()*1e6:.1f},{PV_mean[k250].max()*1e6:.1f}] PVU  "
      f"Event=[{PV_event[k250].min()*1e6:.1f},{PV_event[k250].max()*1e6:.1f}] PVU")

# %% [markdown]
# ## 3. Wu Q Cross-Compare
# %%
def read_wu_ascii(fp):
    d = []; 
    with open(fp) as f:
        for ln in f:
            for tk in ln.split(): d.append(float(tk))
    return np.array(d[:8]), np.array(d[8:])

_, vq = read_wu_ascii(WU_DIR / "event_q.out")
# event_q.out layout after header: [THB(4437), THT(4437), Q_K2..Q_K7(6×4437)]
# 250 hPa = K=7 → interior index 5 → offset = (2 theta blocks + 5 q blocks) × block
q_wu_offset = (2 + 5) * block  # = 7 * 4437 = 31059
Q_wu = vq[q_wu_offset : q_wu_offset + block].reshape(NY, NX)
valid = Q_wu < 9999
if valid.sum():
    # Wu Q is in proprietary units; compare ratio to ERA5 PV (both in SI-equivalent)
    r = Q_wu[valid] / PV_event[k250][valid]
    print(f"Wu Q/ERA5 PV ratio: median={np.median(r):.0f} (Wu Q ≠ PVU — proprietary units)")

# %% [markdown]
# ## 4. Induced Winds from Piecewise ψ′ + Geopotential Φ
# %%
_, vp = read_wu_ascii(WU_DIR / "event_pert.out")
pp = 2 * NW * block  # 1 piece: NW HP (H) + NW SP (ψ)
# --- Parse piecewise geopotential height H [m] → geopotential Φ [m²/s²] ---
HP_raw = np.zeros((NPIECES, NW, NY, NX), dtype=np.float32)
PSI_raw = np.zeros((NPIECES, NW, NY, NX), dtype=np.float32)
for ip in range(NPIECES):
    base = ip * pp
    for k in range(NW):
        HP_raw[ip, k] = vp[base + k*block: base + (k+1)*block].reshape(NY, NX)
    for k in range(NW):
        PSI_raw[ip, k] = vp[base + NW*block + k*block: base + NW*block + (k+1)*block].reshape(NY, NX)

# Convert to SI
PHI_pieces = HP_raw * G                  # geopotential [m²/s²]
PSI_pieces = PSI_raw * PSI_SCALE         # streamfunction [m²/s]

# --- Compute induced winds from ψ gradients ---
RE_WU = 2.0e7 / np.pi
DP = RE_WU * np.radians(DLAT); DL = RE_WU * np.radians(DLON)
AP = np.cos(np.radians(lats))
U_ind = np.zeros((NPIECES, NW, NY, NX), dtype=np.float32)
V_ind = np.zeros((NPIECES, NW, NY, NX), dtype=np.float32)
for ip in range(NPIECES):
    psi = PSI_pieces[ip]  # already in m²/s
    U_ind[ip, :, 1:-1, :] = (psi[:, 2:, :] - psi[:, :-2, :]) / (2.0 * DP)
    for i in range(1, NY-1):
        V_ind[ip, :, i, 1:-1] = (psi[:, i, 2:] - psi[:, i, :-2]) / (2.0 * DL * AP[i])
    for arr in [U_ind, V_ind]:
        arr[ip, :, [0, -1], :] = arr[ip, :, :, [0, -1]] = np.nan

print("Induced winds @250hPa:")
for ip in range(NPIECES):
    ui, vi = U_ind[ip,k250,1:-1,1:-1], V_ind[ip,k250,1:-1,1:-1]
    print(f"  P{ip+1}: U[{ui.min():.0f},{ui.max():.0f}] V[{vi.min():.0f},{vi.max():.0f}] m/s")

# %% [markdown]
# ## 5. Helmholtz ψ Verification
# %%
ds_uh = xr.open_dataset(CLIM_SRC / f"era5_hourly_clim_1990-2020_{EM}_u_helmholtz.nc")
ds_vh = xr.open_dataset(CLIM_SRC / f"era5_hourly_clim_1990-2020_{EM}_v_helmholtz.nc")
ur = ds_uh["u_rot_bar"].sel(day=config.EVENT_DAY, hour=0).squeeze().values
vr = ds_vh["v_rot_bar"].sel(day=config.EVENT_DAY, hour=0).squeeze().values
# Drop 100 hPa — no interp needed
ur = ur[keep_8][:, lat_mask, :][:, :, lon_mask]
vr = vr[keep_8][:, lat_mask, :][:, :, lon_mask]
assert ur.shape[0] == NW, f"Helmholtz shape mismatch: {ur.shape[0]} vs NW={NW}"

dr_lat = np.deg2rad(np.mean(np.abs(np.diff(lats))))
dr_lon = np.deg2rad(np.mean(np.abs(np.diff(lons))))
zeta_rot = np.zeros_like(ur)
for i in range(1, NY-1):
    cl = np.cos(np.deg2rad(lats[i]))
    zeta_rot[:, i, 1:-1] += (vr[:, i, 2:] - vr[:, i, :-2]) / (2.0 * R_E * cl * dr_lon)
zeta_rot[:, 1:-1, :] -= (ur[:, 2:, :] - ur[:, :-2, :]) / (2.0 * R_E * dr_lat)

def poisson_sor(rhs, dx, dy, omega=1.8, niter=5000):
    ny, nx = rhs.shape
    psi = np.zeros_like(rhs)
    d2 = -2/dx**2 - 2/dy**2
    for it in range(niter):
        old = psi.copy()
        psi[1:-1,1:-1] = (1-omega)*psi[1:-1,1:-1] + omega * (
            rhs[1:-1,1:-1]
            - (psi[2:,1:-1]+psi[:-2,1:-1])/dy**2
            - (psi[1:-1,2:]+psi[1:-1,:-2])/dx**2
        ) / d2
        if it % 500 == 0 and np.max(np.abs(psi-old)) < 1e-6:
            break
    return psi

dx_m = R_E * np.cos(np.deg2rad(np.mean(lats))) * dr_lon
dy_m = R_E * dr_lat
psi_rot = np.array([poisson_sor(zeta_rot[k], dx_m, dy_m) for k in range(NW)])  # SI m²/s

# Compare with Wu ψ from meanh (Wu ψ is in per-1e5 units → ×PSI_SCALE)
_, vh = read_wu_ascii(WU_DIR / "meanh")
wu_psi = np.zeros((NW, NY, NX))
for k in range(NW):
    wu_psi[k] = vh[NW*block + k*block: NW*block + (k+1)*block].reshape(NY, NX) * PSI_SCALE

k500 = 3  # 500 hPa is index 3 of 8
ii = (slice(1,-1), slice(1,-1))
rms = np.sqrt(np.mean((psi_rot[k500][ii] - wu_psi[k500][ii])**2))
cor = np.corrcoef(psi_rot[k500][ii].ravel(), wu_psi[k500][ii].ravel())[0,1]
print(f"Helmholtz ψ vs Wu ψ @500hPa: RMS={rms:.2e} m²/s corr={cor:.4f}")

# %% [markdown]
# ## 6. Save Outputs + PV Advection (ALL in SI)
# %%
PLEVS = config.PLEVS  # 8 levels from config

# --- Save piecewise_psi.nc: ψ, geopotential, induced winds ---
ds_psi_out = xr.Dataset({
    "psi":          (["piece","plev","lat","lon"], PSI_pieces.astype(np.float32),
                     {"units": "m2 s-1", "long_name": "streamfunction perturbation"}),
    "geopotential": (["piece","plev","lat","lon"], PHI_pieces.astype(np.float32),
                     {"units": "m2 s-2", "long_name": "geopotential perturbation ($\\Phi = gz$)"}),
    "U_induced":    (["piece","plev","lat","lon"], U_ind.astype(np.float32),
                     {"units": "m s-1", "long_name": "zonal induced wind"}),
    "V_induced":    (["piece","plev","lat","lon"], V_ind.astype(np.float32),
                     {"units": "m s-1", "long_name": "meridional induced wind"}),
}, coords={"piece": np.arange(NPIECES), "plev": PLEVS.astype(np.float32),
           "lat": lats.astype(np.float32), "lon": lons.astype(np.float32)}
).to_netcdf(OUT_DIR / "piecewise_psi.nc")
print("✓ Saved: piecewise_psi.nc (psi [m²/s], geopotential [m²/s²], U/V_induced [m/s])")

# --- Compute PV advection at 250 hPa in SI [K·m²·kg⁻¹·s⁻²] ---
dlat_rad = np.deg2rad(np.mean(np.abs(np.diff(lats))))
dlon_rad = np.deg2rad(np.mean(np.abs(np.diff(lons))))
coslat = np.cos(np.deg2rad(lats))[:, None]
q_anom_250 = PV_event[k250] - PV_mean[k250]  # SI (~1e-6)
dq_dx = np.gradient(q_anom_250, axis=1) / (R_E * coslat * dlon_rad)
dq_dy = np.gradient(q_anom_250, axis=0) / (R_E * dlat_rad)
PVadv = np.zeros((NPIECES, len(lats), len(lons)), dtype=np.float32)
for ip in range(NPIECES):
    PVadv[ip] = -(U_ind[ip, k250] * dq_dx + V_ind[ip, k250] * dq_dy)  # SI s⁻¹ — NO *86400
PVadv[:, [0, -1], :] = np.nan; PVadv[:, :, [0, -1]] = np.nan
for ip in range(NPIECES):
    pv_adv_pvu_day = PVadv[ip] * 86400.0 * 1e6  # convert to PVU/day for display
    print(f"  PV adv P{ip+1}: [{np.nanmin(pv_adv_pvu_day):.1f}, {np.nanmax(pv_adv_pvu_day):.1f}] PVU/day  "
          f"(SI: [{np.nanmin(PVadv[ip]):.2e}, {np.nanmax(PVadv[ip]):.2e}] s⁻¹)")

# --- Save pv_advection.nc ---
ds_adv_out = xr.Dataset({
    "Q_event_250": (["lat","lon"], PV_event[k250].astype(np.float32),
                    {"units": "K m2 kg-1 s-1", "long_name": "event Ertel PV at 250 hPa"}),
    "Q_clim_250":  (["lat","lon"], PV_mean[k250].astype(np.float32),
                    {"units": "K m2 kg-1 s-1", "long_name": "30yr clim Ertel PV at 250 hPa"}),
    "Q_anom_250":  (["lat","lon"], q_anom_250.astype(np.float32),
                    {"units": "K m2 kg-1 s-1", "long_name": "PV anomaly at 250 hPa"}),
    "U_induced_250": (["piece","lat","lon"], U_ind[:, k250].astype(np.float32),
                      {"units": "m s-1", "long_name": "zonal induced wind at 250 hPa"}),
    "V_induced_250": (["piece","lat","lon"], V_ind[:, k250].astype(np.float32),
                      {"units": "m s-1", "long_name": "meridional induced wind at 250 hPa"}),
    "PVadv":        (["piece","lat","lon"], PVadv,
                     {"units": "K m2 kg-1 s-2", "long_name": "PV advection tendency at 250 hPa"}),
}, coords={"piece": np.arange(NPIECES), "lat": lats.astype(np.float32),
           "lon": lons.astype(np.float32)}
).to_netcdf(OUT_DIR / "pv_advection.nc")
print(f"✓ Saved: pv_advection.nc (PVadv [SI s⁻¹], Q_* [SI], U/V_induced [m/s])")
# %% [markdown]
# ## 7. PV Comparison Plot (Wu Q vs ERA5 Ertel PV @ 250 hPa)
# %%
# Display in PVU for readability: multiply stored SI by 1e6
PV_mean_pvu = PV_mean * 1e6   # SI → PVU for display
PV_event_pvu = PV_event * 1e6
q_anom_250_pvu = q_anom_250 * 1e6

proj = ccrs.LambertConformal(central_longitude=-105, central_latitude=50)
pc   = ccrs.PlateCarree()
LON2D, LAT2D = np.meshgrid(lons, lats)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": proj})
titles = ["30yr Clim PV @250 hPa [PVU]", "Event PV @250 hPa [PVU]", "Wu Q (internal units)"]
datas  = [PV_mean_pvu[k250], PV_event_pvu[k250], Q_wu]
cmaps  = ["RdBu_r", "RdBu_r", "RdBu_r"]

for ax, title, data, cmap in zip(axes, titles, datas, cmaps):
    ax.set_extent([config.LON_W-2, config.LON_E+2, config.LAT_S-2, config.LAT_N+2], crs=pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4")
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.2, edgecolor="0.6")
    vm = np.percentile(np.abs(data[data < 9999]), 98) if np.any(data < 9999) else np.percentile(np.abs(data), 98)
    cf = ax.pcolormesh(LON2D, LAT2D, data, cmap=cmap, transform=pc, vmin=-vm, vmax=vm)
    plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02)
    ax.set_title(title, fontsize=10)

plt.suptitle("PV Comparison: 30yr Clim vs Jan 8 2025 Event vs Wu Q\n8 Levels (no interp), 1.5°, IBC=0",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(STEP_DIR / "pv_comparison_250hPa.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved: pv_comparison_250hPa.png")
plt.show()