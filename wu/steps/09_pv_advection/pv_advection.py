# %% [markdown]
# Step 09 — PV Advection by Induced Winds at 250 hPa
#
# Computes PV advection = −U_ind · ∇(Q_anom) at 250 hPa
# for each of the 3 vertical PV pieces. Saves to `pv_advection.nc`
# (updates the file with PVadv field).
#
# %% [markdown]
# ## 1. Load Data
#
# %%
import numpy as np, xarray as xr
from pathlib import Path

import sys
_sys_path_root = str(Path(__file__).resolve().parent.parent.parent)
if _sys_path_root not in sys.path:
    sys.path.insert(0, _sys_path_root)
import config

STEP_DIR = Path(__file__).resolve().parent
OUT_DIR = Path(config.WU_OUT_DIR)

ds = xr.open_dataset(OUT_DIR / "pv_advection.nc")
lats = ds.lat.values
lons = ds.lon.values

Q_anom = ds.Q_anom_250.values      # (51, 87)
U_ind  = ds.U_induced_250.values   # (3, 51, 87)
V_ind  = ds.V_induced_250.values   # (3, 51, 87)
NPIECES = U_ind.shape[0]

print(f"Loaded: U_ind shape={U_ind.shape}, Q_anom shape={Q_anom.shape}")

# %% [markdown]
# ## 2. Compute PV Advection = −(U·∂Q/∂x + V·∂Q/∂y)  [SI: K·m²·kg⁻¹·s⁻²]
#
# %%
R_E = 6.371e6
dlat = np.deg2rad(np.mean(np.abs(np.diff(lats))))
dlon = np.deg2rad(np.mean(np.abs(np.diff(lons))))

# ∂Q/∂x and ∂Q/∂y on the sphere (Q in SI from .nc)
dq_dx = np.gradient(Q_anom, axis=1) / (R_E * np.cos(np.deg2rad(lats))[:, None] * dlon)
dq_dy = np.gradient(Q_anom, axis=0) / (R_E * dlat)

PVadv = np.zeros((NPIECES, len(lats), len(lons)), dtype=np.float32)
for ip in range(NPIECES):
    PVadv[ip] = -(U_ind[ip] * dq_dx + V_ind[ip] * dq_dy)  # SI s⁻¹ — NO *86400

# Mask halos
PVadv[:, [0, -1], :] = np.nan
PVadv[:, :, [0, -1]] = np.nan

print(f"PV advection range per piece (SI s⁻¹ / PVU·day⁻¹):")
PVADV_DAY_SCALE = 1e6 * 86400.0
for ip in range(NPIECES):
    pv = PVadv[ip]
    pv_pvu_day = pv * PVADV_DAY_SCALE
    print(f"  Piece {ip+1}: [{pv_pvu_day.min():.1f}, {pv_pvu_day.max():.1f}] PVU/day  "
          f"p98=±{np.nanpercentile(np.abs(pv_pvu_day), 98):.1f}")

# %% [markdown]
# ## 3. Save Updated pv_advection.nc
#
# %%
# Merge PVadv (SI s⁻¹) into existing dataset
ds_out = ds.assign(
    PVadv=(("piece", "lat", "lon"), PVadv)
)
# Update units attribute
ds_out["PVadv"].attrs = {"units": "K m2 kg-1 s-2", "long_name": "PV advection tendency at 250 hPa"}
ds_out.to_netcdf(OUT_DIR / "pv_advection.nc")
print(f"\n✓ Saved PVadv [SI s⁻¹] to pv_advection.nc")
