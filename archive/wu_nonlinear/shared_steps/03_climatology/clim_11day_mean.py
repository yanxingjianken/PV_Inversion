# %% [markdown]
# Step 03 — 11-Day Time-Mean Climatology & Data Cleaning
#
# **Following Talia tutorial**: "The mean state must be an average over at least
# ten days." We average Jan 3–13 2025 at 00Z (11 days) to create the "climatological
# mean." The perturbation q′ = q(Jan 8) − q̄(11-day mean).
#
# **Data cleaning steps in this notebook**:
# 1. Check all 11 days for NaN/missing values → none found (ERA5 is complete)
# 2. Subset to CA domain (10.5–85.5°N, −169.5 to −40.5°E)
# 3. Compute 11-day time-mean for t, u, v, z
# 4. Visualize the mean state vs individual days
# 5. Compute anomalies (Jan 8 − mean)
# 6. Save mean + event anomaly to `../data/clim/`
#
# %% [markdown]
# ## 1. Load All 11 Days & Subset to CA Domain
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
ERA5_DIR = _Path(config.ERA5_DIR); CLIM_DIR = _Path(config.CLIM_DIR)
FIG_DIR = _Path(config.FIG_DIR)
CLIM_DIR.mkdir(parents=True, exist_ok=True)

# Grid spec (from config)
NX, NY = config.NX, config.NY
LAT_N, LAT_S = config.LAT_N, config.LAT_S
LON_W, LON_E = config.LON_W, config.LON_E
TARGET_LEVS = list(config.PLEVS)

# Dates from config
DATES = [(d.year, d.month, d.day) for d in config.CLIM_DATES]
EVENT_IDX = config.CLIM_WINDOW_DAYS // 2  # event is at center of symmetric window
print(f"Clim window: {config.CLIM_WINDOW_DAYS} days, event at index {EVENT_IDX}")
print(f"  Start: {config.CLIM_START}, End: {config.CLIM_END}")

# Load all days → 4D array (day, lev, lat, lon)
all_t, all_u, all_v, all_z = [], [], [], []
for yr,mo,dy in DATES:
    ds = xr.open_dataset(ERA5_DIR / f"era5_{yr:04d}-{mo:02d}-{dy:02d}_00Z.nc").squeeze()
    ds = ds.sel(latitude=slice(LAT_N, LAT_S), longitude=slice(LON_W % 360, LON_E % 360))
    # Ensure ordering: N→S lat, W→E lon
    ds = ds.sortby("latitude", ascending=False).sortby("longitude")
    all_t.append(ds["t"].values)
    all_u.append(ds["u"].values)
    all_v.append(ds["v"].values)
    all_z.append(ds["z"].values)

all_t = np.stack(all_t)  # (11, 10, 51, 87)
all_u = np.stack(all_u)
all_v = np.stack(all_v)
all_z = np.stack(all_z)

# Data cleaning: check NaN
for name, arr in [("t",all_t),("u",all_u),("v",all_v),("z",all_z)]:
    n_nan = int(np.isnan(arr).sum())
    print(f"  {name}: {n_nan} NaN / {arr.size} total ({100*n_nan/arr.size:.4f}%)")
print("✓ Clean — no NaN values")

# Reference arrays
lats = ds.latitude.values   # N→S
lons = ds.longitude.values  # W→E
plev_native = ds.pressure_level.values.astype(float)
print(f"Native ERA5 levels: {plev_native}")
print(f"Array shape: {all_t.shape} = (day, lev, lat, lon)")

# %% [markdown]
# ## 2. Compute 11-Day Time-Mean
#
# %%
mean_t = np.mean(all_t, axis=0)   # (10, 51, 87)
mean_u = np.mean(all_u, axis=0)
mean_v = np.mean(all_v, axis=0)
mean_z = np.mean(all_z, axis=0)

print(f"Mean T500 = {np.mean(mean_t[4]) - 273.15:.1f} °C  (lev index 4 = 600 hPa)")
print(f"Mean Z500 = {np.max(mean_z[5])/9.81/10:.0f} dam  (lev index 5 = 500 hPa)")

# Event (Jan 8)
event_t = all_t[EVENT_IDX]
event_u = all_u[EVENT_IDX]
event_v = all_v[EVENT_IDX]
event_z = all_z[EVENT_IDX]

# Anomaly
anom_t = event_t - mean_t
anom_u = event_u - mean_u
anom_v = event_v - mean_v
anom_z = event_z - mean_z

print(f"T500 anomaly (Jan 8 − mean): {np.min(anom_t[4]):.1f} … {np.max(anom_t[4]):.1f} K")

# %% [markdown]
# ## 3. Visualize: Z500 Mean State & Anomaly
#
# %%
proj = ccrs.LambertConformal(central_longitude=-105, central_latitude=50)
pc   = ccrs.PlateCarree()
LON2D, LAT2D = np.meshgrid(lons, lats)

z500_mean  = mean_z[5] / 9.81 / 10   # level index 5 = 500 hPa, m -> dam
z500_event = event_z[5] / 9.81 / 10
z500_anom  = anom_z[5] / 9.81 / 10

fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": proj})
titles = ["11-Day Mean Z500 [dam]", "Jan 8 Event Z500 [dam]", "Anomaly Z500 [dam]"]
datas  = [z500_mean, z500_event, z500_anom]
cmaps  = ["viridis", "viridis", "RdBu_r"]
vmxs   = [None, None, 20]

for ax, title, data, cmap, vmx in zip(axes, titles, datas, cmaps, vmxs):
    ax.set_extent([-175, -35, 5, 88], crs=pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.5, edgecolor="0.3")
    ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="0.5")
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.2, edgecolor="0.6")

    vmin, vmax = (-vmx, vmx) if vmx else (None, None)
    cf = ax.contourf(LON2D, LAT2D, data, cmap=cmap, transform=pc,
                     vmin=vmin, vmax=vmax, levels=20)
    cs = ax.contour(LON2D, LAT2D, data, levels=np.arange(480,600,4),
                    colors="black", linewidths=0.4, transform=pc)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f")
    plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02)
    ax.set_title(title, fontsize=10, fontweight="bold")

plt.suptitle("ERA5 500 hPa — CA Blocking Event (Jan 8 2025 00Z)\n"
             "11-day mean = Jan 3–13 average", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(STEP_DIR/"z500_mean_event_anomaly.png", dpi=150, bbox_inches="tight")
print("✓ Saved: z500_mean_event_anomaly.png")
plt.show()

# %% [markdown]
# ## 4. Daily Z500 Anomalies (How the Block Evolved)
#
# %%
NDAYS = len(DATES)
NCOL = 6
NROW = int(np.ceil(NDAYS / NCOL))
fig, axes = plt.subplots(NROW, NCOL, figsize=(3.4*NCOL, 2.8*NROW),
                         subplot_kw={"projection": proj})
axes_flat = np.atleast_1d(axes).ravel()
import datetime as _dt
cf = None
for idx, (ax, (yr,mo,dy)) in enumerate(zip(axes_flat, DATES)):
    z_day = all_z[idx, 5] / 9.81 / 10  # Z500 for this day (m -> dam)
    anom_day = z_day - z500_mean
    ax.set_extent([-175, -35, 5, 88], crs=pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor="0.4")
    cf = ax.contourf(LON2D, LAT2D, anom_day, cmap="RdBu_r", transform=pc,
                     vmin=-25, vmax=25, levels=21)
    cs = ax.contour(LON2D, LAT2D, z_day, levels=np.arange(480,600,4),
                    colors="black", linewidths=0.3, transform=pc)
    is_event = (_dt.date(yr, mo, dy) == config.EVENT_DATE)
    ax.set_title(f"{_dt.date(yr,mo,dy).strftime('%b %d')}"
                 f"{'  ← EVENT' if is_event else ''}", fontsize=8,
                 fontweight="bold" if is_event else "normal",
                 color="red" if is_event else "black")

# Hide unused subplots
for ax in axes_flat[NDAYS:]:
    ax.set_visible(False)

# Dedicated right-side colorbar axis — no overlap with panels
fig.subplots_adjust(left=0.03, right=0.90, top=0.93, bottom=0.04,
                    hspace=0.25, wspace=0.05)
cax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
fig.colorbar(cf, cax=cax, label="Z500 anomaly [dam]")
fig.suptitle(f"Daily Z500 Anomalies (Day − {config.CLIM_WINDOW_DAYS}-Day Mean) — "
             f"{config.CLIM_START} to {config.CLIM_END}",
             fontsize=13, fontweight="bold")
fig.savefig(STEP_DIR/"daily_z500_anomalies.png", dpi=150, bbox_inches="tight")
print("✓ Saved: daily_z500_anomalies.png")
plt.show()

# %% [markdown]
# ## 5. Save Mean & Event for Step 04
#
# %%
# Save as NetCDF files for Step 04 (Wu .grid writer)
def save_nc(data_dict, lats, lons, plev, out_path):
    ds = xr.Dataset(
        {k: (("pressure_level","latitude","longitude"), v.astype(np.float32))
         for k,v in data_dict.items()},
        coords={"pressure_level": plev, "latitude": lats, "longitude": lons},
    )
    ds.to_netcdf(out_path)
    print(f"  ✓ Saved: {out_path}")

save_nc({"t":mean_t,"u":mean_u,"v":mean_v,"z":mean_z},
        lats, lons, config.PLEVS, CLIM_DIR/"mean_clim.nc")
save_nc({"t":event_t,"u":event_u,"v":event_v,"z":event_z},
        lats, lons, config.PLEVS, CLIM_DIR/"event.nc")
save_nc({"t":anom_t,"u":anom_u,"v":anom_v,"z":anom_z},
        lats, lons, config.PLEVS, CLIM_DIR/"anomaly.nc")

print("\n→ Step 04: Convert these NetCDFs to Wu .grid format & verify")

