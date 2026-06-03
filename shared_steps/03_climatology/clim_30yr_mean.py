# %% [markdown]
# Step 03 — 30-Year Climatology (1990–2020) + Event Snapshot (8 levels, no interp)
#
# **Wu levels now match clim levels exactly** (minus 100 hPa):
# [1000, 850, 700, 500, 400, 300, 250, 200] — 8 levels.
# Clim has 9 levels [1000,850,700,500,400,300,250,200,100]; we drop 100 hPa.
#
# **No vertical interpolation needed** — clim and Wu share the same levels.
#
# **Pass A still computes PV** from the mean-state (u,v,t,z) via pvpialln.
# We do NOT feed pre-computed PV to Pass A. The pre-computed 30yr PV from
# `/net/flood/data2/users/x_yan/era/clim/` is used ONLY for step 08 verification.
#
# **Output** (unchanged format for Step 04 compatibility):
# - `mean_clim.nc`: mean t, u, v, z at 8 levels, CA domain
# - `event.nc`: event t, u, v, z at 8 levels, CA domain
#
# %% [markdown]
# ## 1. Configuration & Paths
#
# %%
import numpy as np, xarray as xr, matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from pathlib import Path
import sys

# ---- resolve config (works for both wu/ and wu_nonlinear/) ----
_script_dir = Path(__file__).resolve().parent  # shared_steps/03_climatology/
_project_root = _script_dir.parent.parent       # pv_inversion/ or wu_nonlinear/
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
import config

STEP_DIR = _script_dir
ERA5_DIR = Path(config.ERA5_DIR)
CLIM_DIR = Path(config.CLIM_DIR)
CLIM_DIR.mkdir(parents=True, exist_ok=True)

# ---- 30-year clim source ----
CLIM_SRC = Path("/net/flood/data2/users/x_yan/era/clim")
MONTH_ABBR = {1:"jan",2:"feb",3:"mar",4:"apr",5:"may",6:"jun",
              7:"jul",8:"aug",9:"sep",10:"oct",11:"nov",12:"dec"}
EVENT_MONTH_ABBR = MONTH_ABBR[config.EVENT_MONTH]

# ---- Wu target grid (now 8 levels, same as clim minus 100 hPa) ----
NX, NY = config.NX, config.NY
LAT_N, LAT_S = config.LAT_N, config.LAT_S
LON_W, LON_E = config.LON_W, config.LON_E
WU_LEVS = np.array([1000, 850, 700, 500, 400, 300, 250, 200], dtype=float)
NW = len(WU_LEVS)

# ---- Event date ----
EVENT_DAY = config.EVENT_DAY
EVENT_YR  = config.EVENT_YEAR

print(f"Config: NX={NX}, NY={NY}, Wu levs={list(WU_LEVS)} hPa  (NW={NW})")
print(f"Event: {EVENT_YR}-{config.EVENT_MONTH:02d}-{EVENT_DAY:02d}")
print(f"Clim source: {CLIM_SRC}")

# %% [markdown]
# ## 2. Load 30-Year Climatology (Jan 8, 00Z) — Drop 100 hPa
#
# %%
# Load all four variables from 30yr monthly clim
clim_vars = {}
for var in ["t", "u", "v", "z"]:
    fname = f"era5_hourly_clim_1990-2020_{EVENT_MONTH_ABBR}_{var}.nc"
    ds = xr.open_dataset(CLIM_SRC / fname)
    # Extract day=8, hour=0 (1-indexed day, 0-indexed hour)
    da = ds[var].sel(day=EVENT_DAY, hour=0).squeeze()
    clim_vars[var] = da
    print(f"  Loaded {var}: shape={da.shape}")

# Clim native levels: [1000, 850, 700, 500, 400, 300, 250, 200, 100]
clim_plev = clim_vars["t"].pressure_level.values.astype(float)
clim_lat  = clim_vars["t"].latitude.values.astype(float)
clim_lon  = clim_vars["t"].longitude.values.astype(float)

print(f"\nClim native: {len(clim_plev)} levels = {list(clim_plev)} hPa")

# Drop 100 hPa — keep first 8 levels (exact match with Wu levels)
drop_100 = clim_plev != 100.0
clim_plev_8 = clim_plev[drop_100]
print(f"After dropping 100 hPa: {len(clim_plev_8)} levels = {list(clim_plev_8)} hPa")
assert np.allclose(clim_plev_8, WU_LEVS), f"Level mismatch! clim={list(clim_plev_8)} vs wu={list(WU_LEVS)}"
print("✓ Clim levels match Wu levels exactly — no interpolation needed!")

mean_t = clim_vars["t"].values[drop_100]  # (8, 61, 240)
mean_u = clim_vars["u"].values[drop_100]
mean_v = clim_vars["v"].values[drop_100]
mean_z = clim_vars["z"].values[drop_100]

# %% [markdown]
# ## 3. Subset to CA Domain (1.5° NH → same resolution, no coarsening)
#
# %%
# --- Subset to CA domain ---
lat_mask = (clim_lat <= LAT_N) & (clim_lat >= LAT_S)
_lon_w = LON_W % 360
_lon_e = LON_E % 360
_clim_lon_360 = clim_lon % 360
if _lon_w <= _lon_e:
    lon_mask = (_clim_lon_360 >= _lon_w) & (_clim_lon_360 <= _lon_e)
else:
    lon_mask = (_clim_lon_360 >= _lon_w) | (_clim_lon_360 <= _lon_e)

lats_out = clim_lat[lat_mask]
lons_out = clim_lon[lon_mask]

nlat_sub = int(lat_mask.sum())
nlon_sub = int(lon_mask.sum())
print(f"\nSubset: lat [{lats_out[0]:.1f}..{lats_out[-1]:.1f}] ({nlat_sub} pts)")
print(f"        lon [{lons_out[0]:.1f}..{lons_out[-1]:.1f}] ({nlon_sub} pts)")
assert nlat_sub == NY and nlon_sub == NX, \
    f"Grid mismatch! Subset ({nlat_sub},{nlon_sub}) vs config ({NY},{NX})"

mean_t = mean_t[:, lat_mask, :][:, :, lon_mask]
mean_u = mean_u[:, lat_mask, :][:, :, lon_mask]
mean_v = mean_v[:, lat_mask, :][:, :, lon_mask]
mean_z = mean_z[:, lat_mask, :][:, :, lon_mask]

print(f"Mean state shape: t={mean_t.shape}")

# %% [markdown]
# ## 4. Load Event Data (Jan 8 00Z) — subset to 8 levels (no re-download needed)
#
# Existing ERA5 data has 10 levels. We subset to the same 8 levels as the clim.
# Future downloads will use 8 levels directly (see shared_steps/01_download/).
#
# %%
event_fname = f"era5_{EVENT_YR:04d}-{config.EVENT_MONTH:02d}-{EVENT_DAY:02d}_00Z.nc"
event_path = ERA5_DIR / event_fname
if not event_path.exists():
    raise FileNotFoundError(
        f"Event file {event_fname} not found in {ERA5_DIR}.\n"
        f"Run shared_steps/01_download/download_era5.py first.")

print(f"Event file: {event_path.name}")
ds_ev = xr.open_dataset(event_path).squeeze()

ev_plev = ds_ev.pressure_level.values.astype(float)
ev_lat  = ds_ev.latitude.values.astype(float)
ev_lon  = ds_ev.longitude.values.astype(float)

# Sort lat descending (N→S) and lon ascending (W→E)
ds_ev = ds_ev.sortby("latitude", ascending=False).sortby("longitude")

# Subset to 8 Wu levels from existing 10-level data
ev_keep = np.isin(ev_plev, WU_LEVS)
ev_plev_8 = ev_plev[ev_keep]
print(f"Event native levels: {list(ev_plev)} → kept {len(ev_plev_8)}: {list(ev_plev_8)}")
assert np.allclose(ev_plev_8, WU_LEVS), \
    f"Event level mismatch! {list(ev_plev_8)} vs {list(WU_LEVS)}"

event_t = ds_ev["t"].values[ev_keep]
event_u = ds_ev["u"].values[ev_keep]
event_v = ds_ev["v"].values[ev_keep]
event_z = ds_ev["z"].values[ev_keep]

# Subset to CA domain (handle 0-360 lon convention)
ev_lat_mask = (ds_ev.latitude.values <= LAT_N) & (ds_ev.latitude.values >= LAT_S)
_ev_lon_360 = ds_ev.longitude.values % 360
if _lon_w <= _lon_e:
    ev_lon_mask = (_ev_lon_360 >= _lon_w) & (_ev_lon_360 <= _lon_e)
else:
    ev_lon_mask = (_ev_lon_360 >= _lon_w) | (_ev_lon_360 <= _lon_e)

event_t = event_t[:, ev_lat_mask, :][:, :, ev_lon_mask]
event_u = event_u[:, ev_lat_mask, :][:, :, ev_lon_mask]
event_v = event_v[:, ev_lat_mask, :][:, :, ev_lon_mask]
event_z = event_z[:, ev_lat_mask, :][:, :, ev_lon_mask]

print(f"Event shape: t={event_t.shape}")

# %% [markdown]
# ## 5. Save mean_clim.nc & event.nc (Step 04 compatible)
#
# %%
def save_nc(data_dict, lats_arr, lons_arr, plev_arr, out_path):
    ds = xr.Dataset(
        {k: (("pressure_level","latitude","longitude"), v.astype(np.float32))
         for k, v in data_dict.items()},
        coords={"pressure_level": plev_arr, "latitude": lats_arr, "longitude": lons_arr},
    )
    ds.to_netcdf(out_path)
    print(f"  ✓ Saved: {out_path} ({out_path.stat().st_size // 1024} KB)")

save_nc({"t":mean_t, "u":mean_u, "v":mean_v, "z":mean_z},
        lats_out, lons_out, WU_LEVS, CLIM_DIR / "mean_clim.nc")
save_nc({"t":event_t, "u":event_u, "v":event_v, "z":event_z},
        lats_out, lons_out, WU_LEVS, CLIM_DIR / "event.nc")

# %% [markdown]
# ## 6. Quick Diagnostic: Z500 Mean State & Anomaly
#
# %%
proj = ccrs.LambertConformal(central_longitude=-105, central_latitude=50)
pc   = ccrs.PlateCarree()
LON2D, LAT2D = np.meshgrid(lons_out, lats_out)

z500_mean  = mean_z[3] / 9.81 / 10   # idx 3 = 500 hPa, m²/s² → dam
z500_event = event_z[3] / 9.81 / 10
z500_anom  = z500_event - z500_mean

fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": proj})
titles = ["30yr Clim Mean Z500 [dam]", f"Jan 8 {EVENT_YR} Event Z500 [dam]", "Anomaly Z500 [dam]"]
datas  = [z500_mean, z500_event, z500_anom]
cmaps  = ["viridis", "viridis", "RdBu_r"]

for ax, title, data, cmap in zip(axes, titles, datas, cmaps):
    ax.set_extent([LON_W-2, LON_E+2, LAT_S-2, LAT_N+2], crs=pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.5, edgecolor="0.3")
    ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="0.5")
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.2, edgecolor="0.6")
    vmax = max(abs(data.min()), abs(data.max())) if "Anom" in title else None
    cf = ax.contourf(LON2D, LAT2D, data, cmap=cmap, transform=pc,
                     vmin=(-vmax if vmax else None), vmax=vmax, levels=20)
    cs = ax.contour(LON2D, LAT2D, data, levels=np.arange(460,610,4),
                    colors="black", linewidths=0.4, transform=pc)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f")
    plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02)
    ax.set_title(title, fontsize=10, fontweight="bold")

plt.suptitle(f"ERA5 500 hPa — CA Blocking Event ({EVENT_YR}-{config.EVENT_MONTH:02d}-{EVENT_DAY:02d} 00Z)\n"
             f"Mean = 30-year (1990–2020) {EVENT_MONTH_ABBR.title()} climatology",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(STEP_DIR / "z500_30yr_clim_event_anomaly.png", dpi=150, bbox_inches="tight")
print(f"\n✓ Saved: z500_30yr_clim_event_anomaly.png")
plt.show()

print("\n=== Step 03 complete ===")
print(f"  mean_clim.nc: {CLIM_DIR / 'mean_clim.nc'}")
print(f"  event.nc:     {CLIM_DIR / 'event.nc'}")
print(f"  Wu levels (no interp): {list(WU_LEVS)} hPa  ({NW} levels)")
print("  Ready for Step 04 (write .grid files).")
