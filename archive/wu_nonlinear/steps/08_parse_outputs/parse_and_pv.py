# %% [markdown]
# Step 08 — Parse Outputs + Pre-Computed PV + Helmholtz ψ
#
# Uses 30-year (1990-2020) monthly PV climatology instead of computing Ertel PV.
# Also loads Helmholtz rotational wind for ψ verification.
#
# %%
import numpy as np, xarray as xr, matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from pathlib import Path
from scipy.interpolate import interp1d

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

# %% [markdown]
# ## 1. Vertical Interpolation (log-p, ascending x for interp1d)
# %%
def vert_interp_logp(data_9lev, p_src_hPa, p_tgt_hPa):
    p_asc = p_src_hPa[::-1]
    logp_src = np.log(p_asc)
    logp_tgt = np.log(p_tgt_hPa)
    n_src = len(p_src_hPa)
    data_asc = data_9lev[::-1]
    flat = data_asc.reshape(n_src, -1)
    interp = interp1d(logp_src, flat, axis=0, kind="linear",
                      bounds_error=False, fill_value="extrapolate")
    flat_tgt = interp(logp_tgt)
    return flat_tgt.reshape((len(p_tgt_hPa),) + data_9lev.shape[1:])

# %% [markdown]
# ## 2. Load Pre-Computed 30yr PV Climatology
# %%
ds_mean = xr.open_dataset(CLIM_DIR / "mean_clim.nc")
ds_event = xr.open_dataset(CLIM_DIR / "event.nc")
lats = ds_mean.latitude.values
lons = ds_mean.longitude.values
WU_LEVS = ds_mean.pressure_level.values.astype(float)

ds_pv = xr.open_dataset(CLIM_SRC / f"era5_hourly_clim_1990-2020_{EM}_pv.nc")
pv_raw = ds_pv["pv"].sel(day=config.EVENT_DAY, hour=0).squeeze().values
clim_plev = ds_pv.pressure_level.values.astype(float)
clim_lat  = ds_pv.latitude.values.astype(float)
clim_lon  = ds_pv.longitude.values.astype(float)

PV_mean_full = vert_interp_logp(pv_raw, clim_plev, WU_LEVS) * 1.0e6

lat_mask = (clim_lat <= config.LAT_N) & (clim_lat >= config.LAT_S)
_lw, _le = config.LON_W % 360, config.LON_E % 360
_clon = clim_lon % 360
lon_mask = (_clon >= _lw) & (_clon <= _le)
PV_mean = PV_mean_full[:, lat_mask, :][:, :, lon_mask]

# Event PV from t,u,v (no pre-computed event PV)
t_ev, u_ev, v_ev = ds_event["t"].values, ds_event["u"].values, ds_event["v"].values
plev_Pa = WU_LEVS * 100.0

def ertel_pv(t3d, u3d, v3d, plev_Pa_arr, lat_arr):
    P0, Rd, Cp = 1.0e5, 287.0, 1004.0; R_E = 6.371e6
    theta = t3d * (P0 / plev_Pa_arr[:, None, None]) ** (Rd / Cp)
    dth = np.gradient(theta, plev_Pa_arr, axis=0)
    dla = np.deg2rad(np.gradient(lat_arr))[None, :, None]
    dlo = np.deg2rad(np.gradient(lons))[None, None, :]
    cl = np.cos(np.deg2rad(lat_arr))[None, :, None]
    zeta = (np.gradient(v3d, axis=2) / (R_E * cl * dlo)
          - np.gradient(u3d, axis=1) / (R_E * dla))
    f = 2 * 7.292e-5 * np.sin(np.deg2rad(lat_arr))[None, :, None]
    return -9.81 * (zeta + f) * dth * 1.0e6

PV_event = ertel_pv(t_ev, u_ev, v_ev, plev_Pa, lats)

k250 = 8
print(f"PV @250hPa: Mean(30yr)=[{PV_mean[k250].min():.1f},{PV_mean[k250].max():.1f}] PVU  "
      f"Event=[{PV_event[k250].min():.1f},{PV_event[k250].max():.1f}] PVU")

# %% [markdown]
# ## 3. Wu Q Cross-Compare
# %%
def read_wu_ascii(fp):
    """Read F10.2 Fortran output: split every 10 chars (robust to overflow)."""
    d = []
    with open(fp) as f:
        for line in f:
            line = line.rstrip('\n')
            for i in range(0, len(line), 10):
                tok = line[i:i+10].strip()
                if tok:
                    try:
                        d.append(float(tok))
                    except ValueError:
                        d.append(np.nan)
    hdr = np.array(d[:8]) if len(d) >= 8 else np.zeros(8)
    return hdr, np.array(d[8:])

_, vq = read_wu_ascii(WU_DIR / "event_q.out")
Q_wu = vq[(2+7)*block:(3+7)*block].reshape(NY, NX)
valid = Q_wu < 9999
if valid.sum():
    r = Q_wu[valid] / PV_event[k250][valid]
    print(f"Wu Q/ERA5 PV ratio: median={np.median(r):.0f}")

# %% [markdown]
# ## 4. Induced Winds from Piecewise ψ′
# %%
_, vp = read_wu_ascii(WU_DIR / "event_pert.out")
pp = 2 * NW * block
SP = np.zeros((NPIECES, NW, NY, NX))
for ip in range(NPIECES):
    base = ip * pp + NW * block
    for k in range(NW):
        SP[ip, k] = vp[base + k*block: base + (k+1)*block].reshape(NY, NX)

RE = 2.0e7 / np.pi
DP = RE * np.radians(DLAT); DL = RE * np.radians(DLON)
AP = np.cos(np.radians(lats))
U_ind = np.zeros((NPIECES, NW, NY, NX))
V_ind = np.zeros((NPIECES, NW, NY, NX))
for ip in range(NPIECES):
    psi = SP[ip] * 1.0e5
    U_ind[ip, :, 1:-1, :] = (psi[:, 2:, :] - psi[:, :-2, :]) / (2.0 * DP)
    for i in range(1, NY-1):
        V_ind[ip, :, i, 1:-1] = (psi[:, i, 2:] - psi[:, i, :-2]) / (2.0 * DL * AP[i])
    for arr in [U_ind, V_ind]:
        arr[ip, :, [0, -1], :] = arr[ip, :, :, [0, -1]] = np.nan

print("Induced winds @250hPa:")
for ip in range(NPIECES):
    ui, vi = U_ind[ip,8,1:-1,1:-1], V_ind[ip,8,1:-1,1:-1]
    print(f"  P{ip+1}: U[{ui.min():.0f},{ui.max():.0f}] V[{vi.min():.0f},{vi.max():.0f}] m/s")

# %% [markdown]
# ## 5. Helmholtz ψ Verification
# %%
ds_uh = xr.open_dataset(CLIM_SRC / f"era5_hourly_clim_1990-2020_{EM}_u_helmholtz.nc")
ds_vh = xr.open_dataset(CLIM_SRC / f"era5_hourly_clim_1990-2020_{EM}_v_helmholtz.nc")
ur = ds_uh["u_rot_bar"].sel(day=config.EVENT_DAY, hour=0).squeeze().values
vr = ds_vh["v_rot_bar"].sel(day=config.EVENT_DAY, hour=0).squeeze().values
ur = vert_interp_logp(ur, clim_plev, WU_LEVS)[:, lat_mask, :][:, :, lon_mask]
vr = vert_interp_logp(vr, clim_plev, WU_LEVS)[:, lat_mask, :][:, :, lon_mask]

R_E = 6.371e6
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
psi_rot = np.array([poisson_sor(zeta_rot[k], dx_m, dy_m) for k in range(NW)])
psi_rot_s = psi_rot / 1.0e5

_, vh = read_wu_ascii(WU_DIR / "meanh")
wu_psi = np.zeros((NW, NY, NX))
for k in range(NW):
    wu_psi[k] = vh[NW*block + k*block: NW*block + (k+1)*block].reshape(NY, NX)

k5 = 5
ii = (slice(1,-1), slice(1,-1))
rms = np.sqrt(np.mean((psi_rot_s[k5][ii] - wu_psi[k5][ii])**2))
cor = np.corrcoef(psi_rot_s[k5][ii].ravel(), wu_psi[k5][ii].ravel())[0,1]
print(f"Helmholtz ψ vs Wu ψ @500hPa: RMS={rms:.1f} corr={cor:.4f}")

# %% [markdown]
# ## 6. Save Outputs
# %%
PLEVS = np.array([1000.,925.,850.,700.,600.,500.,400.,300.,250.,200.])
xr.Dataset({
    "HP": (["piece","plev","lat","lon"], SP.astype(np.float32)),
    "U_induced": (["piece","plev","lat","lon"], U_ind.astype(np.float32)),
    "V_induced": (["piece","plev","lat","lon"], V_ind.astype(np.float32)),
}, coords={"piece": np.arange(NPIECES), "plev": PLEVS, "lat": lats, "lon": lons}
).to_netcdf(OUT_DIR / "piecewise_psi.nc")

xr.Dataset({
    "Q_event_250": (["lat","lon"], PV_event[k250].astype(np.float32)),
    "Q_clim_250":  (["lat","lon"], PV_mean[k250].astype(np.float32)),
    "Q_anom_250":  (["lat","lon"], (PV_event[k250]-PV_mean[k250]).astype(np.float32)),
    "U_induced_250": (["piece","lat","lon"], U_ind[:, 8].astype(np.float32)),
    "V_induced_250": (["piece","lat","lon"], V_ind[:, 8].astype(np.float32)),
}, coords={"piece": np.arange(NPIECES), "lat": lats, "lon": lons}
).to_netcdf(OUT_DIR / "pv_advection.nc")
print(f"\n✓ Saved: piecewise_psi.nc, pv_advection.nc")
