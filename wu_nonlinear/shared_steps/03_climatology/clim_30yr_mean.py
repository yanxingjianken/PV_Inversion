# %% [markdown]
# Step 03 — 30-Year Climatology (1990–2020) + Event Snapshot
#
# **Replaces** the 11-day running-mean approach. Uses pre-computed
# 30-year (1990–2020) monthly climatologies from
# `/net/flood/data2/users/x_yan/era/clim/`.
#
# **Key changes from old step 03:**
# - Mean state = 30-year Jan climatology (day=8, hour=0)
# - No 11-day download needed (saves ~10 CDS calls)
# - Vertical interpolation: clim has 9 levels [1000,850,700,500,400,300,250,200,100]
#   → Wu needs 10 levels [1000,925,850,700,600,500,400,300,250,200]
# - Helmholtz rotational wind available for optional ψ verification
# - Event data: reuses existing download or loads from era5/ directory
#
# **Output** (unchanged format for Step 04 compatibility):
# - `mean_clim.nc`: mean t, u, v, z at Wu's 10 levels, CA domain
# - `event.nc`: event t, u, v, z at Wu's 10 levels, CA domain
#
# %% [markdown]
# ## 1. Configuration & Paths
#
# %%
import numpy as np, xarray as xr, matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from pathlib import Path
from scipy.interpolate import interp1d
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
FIG_DIR  = Path(config.FIG_DIR)
CLIM_DIR.mkdir(parents=True, exist_ok=True)

# ---- 30-year clim source ----
CLIM_SRC = Path("/net/flood/data2/users/x_yan/era/clim")
MONTH_ABBR = {1:"jan",2:"feb",3:"mar",4:"apr",5:"may",6:"jun",
              7:"jul",8:"aug",9:"sep",10:"oct",11:"nov",12:"dec"}
EVENT_MONTH_ABBR = MONTH_ABBR[config.EVENT_MONTH]

# ---- Wu target grid ----
NX, NY = config.NX, config.NY
LAT_N, LAT_S = config.LAT_N, config.LAT_S
LON_W, LON_E = config.LON_W, config.LON_E
WU_LEVS = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200], dtype=float)
NW = len(WU_LEVS)

# ---- Event date ----
EVENT_DAY = config.EVENT_DAY
EVENT_YR  = config.EVENT_YEAR

print(f"Config: NX={NX}, NY={NY}, Wu levs={list(WU_LEVS)} hPa")
print(f"Event: {EVENT_YR}-{config.EVENT_MONTH:02d}-{EVENT_DAY:02d}")
print(f"Clim source: {CLIM_SRC}")

# %% [markdown]
# ## 2. Load 30-Year Climatology (Jan 8, 00Z)
#
# %%
# Load all four variables from 30yr monthly clim
clim_vars = {}
for var in ["t", "u", "v", "z"]:
    fname = f"era5_hourly_clim_1990-2020_{EVENT_MONTH_ABBR}_{var}.nc"
    ds = xr.open_dataset(CLIM_SRC / fname)
    # Extract day=8, hour=0 (1-indexed day, 0-indexed hour)
    # Clim dims: (month, day, hour, pressure_level, latitude, longitude)
    da = ds[var].sel(day=EVENT_DAY, hour=0).squeeze()
    clim_vars[var] = da
    print(f"  Loaded {var}: shape={da.shape}, levs={list(da.pressure_level.values)}")

# Get coordinates from clim
clim_plev = clim_vars["t"].pressure_level.values.astype(float)  # [1000,850,700,500,400,300,250,200,100]
clim_lat  = clim_vars["t"].latitude.values.astype(float)        # 90..0 N→S
clim_lon  = clim_vars["t"].longitude.values.astype(float)       # -180..178.5

print(f"\nClim native: {len(clim_plev)} levels = {list(clim_plev)} hPa")
print(f"Clim native: lat [{clim_lat[0]:.1f}..{clim_lat[-1]:.1f}] ({len(clim_lat)} pts)")
print(f"Clim native: lon [{clim_lon[0]:.1f}..{clim_lon[-1]:.1f}] ({len(clim_lon)} pts)")

# %% [markdown]
# ## 3. Vertical Interpolation: 9 Clim Levels → 10 Wu Levels
#
# Log-pressure interpolation (standard for atmospheric vertical interpolation).
#
# %%
def vert_interp_logp(data_9lev, p_src_hPa, p_tgt_hPa):
    """Interpolate (..., nlev_src, nlat, nlon) to (..., nlev_tgt, nlat, nlon)
    in log-pressure space. Linear in log(p).
    interp1d requires strictly increasing x → reverse source to ascending p.
    Output is in the order of p_tgt_hPa."""
    p_asc = p_src_hPa[::-1]
    logp_src = np.log(p_asc)
    logp_tgt = np.log(p_tgt_hPa)
    n_src = len(p_src_hPa)
    data_asc = data_9lev[::-1]  # reverse to ascending p
    flat = data_asc.reshape(n_src, -1)
    interp = interp1d(logp_src, flat, axis=0, kind="linear",
                      bounds_error=False, fill_value="extrapolate")
    flat_tgt = interp(logp_tgt)
    shape_out = (len(p_tgt_hPa),) + data_9lev.shape[1:]
    return flat_tgt.reshape(shape_out)

mean_t = vert_interp_logp(clim_vars["t"].values, clim_plev, WU_LEVS)
mean_u = vert_interp_logp(clim_vars["u"].values, clim_plev, WU_LEVS)
mean_v = vert_interp_logp(clim_vars["v"].values, clim_plev, WU_LEVS)
mean_z = vert_interp_logp(clim_vars["z"].values, clim_plev, WU_LEVS)

print(f"\nAfter vertical interp: mean_t shape={mean_t.shape}")
print(f"  Wu levels: {list(WU_LEVS)} hPa")
print(f"  T500 (clim) = {np.mean(mean_t[5]):.1f} K  (idx 5 = 500 hPa)")

# %% [markdown]
# ## 4. Subset to CA Domain + Coarsen if Needed
#
# Clim is 1.5° NH (61 lat × 240 lon). Wu target may be 1.5° or 2.5°.
# Strategy: subset to domain then coarsen if resolutions differ.
#
# %%
# --- Subset to CA domain ---
# Latitude: N→S (90→0 in clim)
lat_mask = (clim_lat <= LAT_N) & (clim_lat >= LAT_S)
# Longitude: robust across both -180..180 and 0..360 conventions
_lon_w = LON_W % 360
_lon_e = LON_E % 360
# Convert clim lon to 0..360 for comparison, then mask
_clim_lon_360 = clim_lon % 360
if _lon_w <= _lon_e:
    lon_mask = (_clim_lon_360 >= _lon_w) & (_clim_lon_360 <= _lon_e)
else:
    # Wraparound (e.g., 350 to 10)
    lon_mask = (_clim_lon_360 >= _lon_w) | (_clim_lon_360 <= _lon_e)

lats_subset = clim_lat[lat_mask]
lons_subset = clim_lon[lon_mask]

nlat_sub = int(lat_mask.sum())
nlon_sub = int(lon_mask.sum())
print(f"\nSubset: lat [{lats_subset[0]:.1f}..{lats_subset[-1]:.1f}] ({nlat_sub} pts)")
print(f"        lon [{lons_subset[0]:.1f}..{lons_subset[-1]:.1f}] ({nlon_sub} pts)")

mean_t = mean_t[:, lat_mask, :][:, :, lon_mask]
mean_u = mean_u[:, lat_mask, :][:, :, lon_mask]
mean_v = mean_v[:, lat_mask, :][:, :, lon_mask]
mean_z = mean_z[:, lat_mask, :][:, :, lon_mask]

# --- Coarsen if needed (for wu_nonlinear 2.5°) ---
clim_dlat = np.abs(np.mean(np.diff(lats_subset)))
clim_dlon = np.abs(np.mean(np.diff(lons_subset)))

factor_lat, factor_lon = 1, 1
lats_out, lons_out = lats_subset, lons_subset

# Target grid (used by both mean and event regridding)
lats_tgt = np.arange(config.LAT_N, config.LAT_S - 0.001, -config.DLAT)
lons_tgt = np.arange(config.LON_W, config.LON_E + 0.001, config.DLON)

def regrid_to_target(data_3d, lats_src, lons_src):
    """Regrid 3D (plev, lat, lon) data to target grid via linear interp."""
    da = xr.DataArray(
        data_3d, dims=["plev","lat","lon"],
        coords={"plev": WU_LEVS, "lat": lats_src, "lon": lons_src}
    )
    return da.interp(lat=lats_tgt, lon=lons_tgt, method="linear").values

if abs(clim_dlat - config.DLAT) > 0.01 or abs(clim_dlon - config.DLON) > 0.01:
    print(f"\nRegridding: {clim_dlat:.1f}°→{config.DLAT:.1f}°")
    print(f"  Target: lat [{lats_tgt[0]:.1f}..{lats_tgt[-1]:.1f}] ({len(lats_tgt)} pts)")
    print(f"          lon [{lons_tgt[0]:.1f}..{lons_tgt[-1]:.1f}] ({len(lons_tgt)} pts)")

    mean_t = regrid_to_target(mean_t, lats_subset, lons_subset)
    mean_u = regrid_to_target(mean_u, lats_subset, lons_subset)
    mean_v = regrid_to_target(mean_v, lats_subset, lons_subset)
    mean_z = regrid_to_target(mean_z, lats_subset, lons_subset)
    lats_out, lons_out = lats_tgt, lons_tgt
    print(f"  Regridded mean: shape={mean_t.shape}")

print(f"Final mean: t shape={mean_t.shape}, lat={len(lats_out)}, lon={len(lons_out)}")

# %% [markdown]
# ## 5. Load Event Data
#
# Reuse existing event download from ERA5_DIR.
# If event data missing, fall back to clim for the mean state.
#
# %%
EVENT_DS_PATHS = sorted(ERA5_DIR.glob(f"era5_{EVENT_YR:04d}-{config.EVENT_MONTH:02d}-*.nc"))
print(f"\nEvent ERA5 files found: {len(EVENT_DS_PATHS)}")
for p in EVENT_DS_PATHS:
    print(f"  {p.name}")

if len(EVENT_DS_PATHS) == 0:
    print("\n⚠️  No event ERA5 data found. Will create event.nc = mean (zero anomaly).")
    print("    Re-run Step 01 to download event data, then re-run this step.")
    event_t, event_u, event_v, event_z = mean_t, mean_u, mean_v, mean_z
else:
    # Find the specific event date file
    event_fname = f"era5_{EVENT_YR:04d}-{config.EVENT_MONTH:02d}-{EVENT_DAY:02d}_00Z.nc"
    event_path = ERA5_DIR / event_fname
    if not event_path.exists():
        print(f"\n⚠️  Event file {event_fname} not found. Using first available.")
        event_path = EVENT_DS_PATHS[0]
    print(f"  Event file: {event_path.name}")
    ds_ev = xr.open_dataset(event_path).squeeze()

    # Interpolate to Wu pressure levels if needed
    ev_plev = ds_ev.pressure_level.values.astype(float)
    ev_lat  = ds_ev.latitude.values.astype(float)
    ev_lon  = ds_ev.longitude.values.astype(float)

    # Sort lat descending (N→S) and lon ascending (W→E)
    ds_ev = ds_ev.sortby("latitude", ascending=False).sortby("longitude")

    # Interpolate vertically if needed
    if not np.allclose(ev_plev, WU_LEVS, atol=0.01):
        print(f"  Event levels {list(ev_plev)} → Wu levels {list(WU_LEVS)}")
        event_t = vert_interp_logp(ds_ev["t"].values, ev_plev, WU_LEVS)
        event_u = vert_interp_logp(ds_ev["u"].values, ev_plev, WU_LEVS)
        event_v = vert_interp_logp(ds_ev["v"].values, ev_plev, WU_LEVS)
        event_z = vert_interp_logp(ds_ev["z"].values, ev_plev, WU_LEVS)
    else:
        event_t = ds_ev["t"].values
        event_u = ds_ev["u"].values
        event_v = ds_ev["v"].values
        event_z = ds_ev["z"].values

    # Subset to CA domain (handle 0-360 lon convention)
    ev_lat_mask = (ds_ev.latitude.values <= LAT_N) & (ds_ev.latitude.values >= LAT_S)
    _ev_lon_360 = ds_ev.longitude.values % 360
    if _lon_w <= _lon_e:
        ev_lon_mask = (_ev_lon_360 >= _lon_w) & (_ev_lon_360 <= _lon_e)
    else:
        ev_lon_mask = (_ev_lon_360 >= _lon_w) | (_ev_lon_360 <= _lon_e)
    ev_lats_sub = ds_ev.latitude.values[ev_lat_mask]
    ev_lons_sub_raw = ds_ev.longitude.values[ev_lon_mask]
    # Convert to -180..180 for consistency with target grid
    ev_lons_sub = np.where(ev_lons_sub_raw > 180, ev_lons_sub_raw - 360, ev_lons_sub_raw)
    event_t = event_t[:, ev_lat_mask, :][:, :, ev_lon_mask]
    event_u = event_u[:, ev_lat_mask, :][:, :, ev_lon_mask]
    event_v = event_v[:, ev_lat_mask, :][:, :, ev_lon_mask]
    event_z = event_z[:, ev_lat_mask, :][:, :, ev_lon_mask]

    # Regrid if event resolution differs from target (e.g., 1.5→2.5)
    if abs(clim_dlat - config.DLAT) > 0.01:
        event_t = regrid_to_target(event_t, ev_lats_sub, ev_lons_sub)
        event_u = regrid_to_target(event_u, ev_lats_sub, ev_lons_sub)
        event_v = regrid_to_target(event_v, ev_lats_sub, ev_lons_sub)
        event_z = regrid_to_target(event_z, ev_lats_sub, ev_lons_sub)

    print(f"  Event shape: t={event_t.shape}")

# %% [markdown]
# ## 6. Save mean_clim.nc & event.nc (Step 04 compatible)
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
# ## 7. Quick Diagnostic: Z500 Mean State & Anomaly
#
# %%
proj = ccrs.LambertConformal(central_longitude=-105, central_latitude=50)
pc   = ccrs.PlateCarree()
LON2D, LAT2D = np.meshgrid(lons_out, lats_out)

z500_mean  = mean_z[5] / 9.81 / 10   # idx 5 = 500 hPa, m²/s² → dam
z500_event = event_z[5] / 9.81 / 10
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
print("  Ready for Step 04 (write .grid files).")
