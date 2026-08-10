# %% [markdown]
# Step 01 — Download ERA5 Event Snapshot (single hour) + Polar Map Viz
#
# The climatology is now the **pre-computed 30-year (1990–2020) hourly clim**
# (`shared_steps/03_climatology/clim_30yr_mean.py` reads it from
# `/net/flood/data2/users/x_yan/era/clim/`), so **NO climatology-window download
# is needed here**. The ONLY ERA5 file this step must fetch is the single event
# hour (`config.EVENT_DATE` 00Z) used as the perturbation state.
#
# **Output**: `data/era5/era5_YYYY-MM-DD_00Z.nc` (1 file, the event hour)

# %%
import cdsapi, os, numpy as np, xarray as xr
import matplotlib.pyplot as plt, cartopy.crs as ccrs, cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from pathlib import Path

import sys; from pathlib import Path as _Path
_sys_path_root = str(_Path(__file__).resolve().parent.parent.parent)
if _sys_path_root not in sys.path: sys.path.insert(0, _sys_path_root)
import config
STEP_DIR = _Path(__file__).resolve().parent
ERA5_DIR = _Path(config.ERA5_DIR); ERA5_DIR.mkdir(parents=True, exist_ok=True)

# Derive from config so the download always matches the inversion grid (8 levels, 1000…200).
LEVELS = [str(int(p)) for p in config.PLEVS]  # 1000,850,700,500,400,300,250,200
VARS   = ["temperature","u_component_of_wind","v_component_of_wind","geopotential"]

# Only the single event hour is needed (climatology is pre-computed elsewhere).
DATES = [(config.EVENT_YEAR, config.EVENT_MONTH, config.EVENT_DAY)]
print(f"Downloading {len(DATES)} event hour to {ERA5_DIR}")
print(f"  Event: {config.EVENT_DATE} 00Z")

# %% [markdown]
# ## 2. Download
#
# %%
c = cdsapi.Client()
for yr,mo,dy in DATES:
    fp = ERA5_DIR / f"era5_{yr:04d}-{mo:02d}-{dy:02d}_00Z.nc"
    if fp.exists():
        print(f"  ✓ {fp.name} ({fp.stat().st_size/1e6:.1f} MB)")
        continue
    print(f"  ↓ {yr}-{mo:02d}-{dy:02d} …", end=" ", flush=True)
    c.retrieve("reanalysis-era5-pressure-levels", {
        "product_type":"reanalysis","format":"netcdf","variable":VARS,
        "pressure_level":LEVELS,"year":str(yr),"month":f"{mo:02d}",
        "day":f"{dy:02d}","time":"00:00","area":[90,0,0,360],"grid":["1.5","1.5"],
    }, str(fp))
    print(f"done ({fp.stat().st_size/1e6:.1f} MB)")
print(f"\n✓ {len(list(ERA5_DIR.glob('*.nc')))} files")

# %% [markdown]
# ## 3. Inspect Event Snapshot
#
# %%
for yr,mo,dy in DATES:
    ds = xr.open_dataset(ERA5_DIR / f"era5_{yr:04d}-{mo:02d}-{dy:02d}_00Z.nc").squeeze()
    t500 = ds["t"].sel(pressure_level=500).values
    z500 = ds["z"].sel(pressure_level=500).values / 9.81
    u250 = ds["u"].sel(pressure_level=250).values
    nn = int(np.isnan(t500).sum())
    print(f"  {yr}-{mo:02d}-{dy:02d}: T500 μ={np.nanmean(t500)-273.15:.1f}°C, "
          f"Z500 max={np.nanmax(z500):.0f} dam, U250 max={np.nanmax(u250):.0f} m/s, NaN={nn}")

# %% [markdown]
# ## 4. Polar Map — Event Z500 (single hour)
#
# A cyclic point is appended at 360° so the filled contours close seamlessly
# across the 0°/360° (Greenwich) meridian — otherwise a white wedge ("spike")
# appears there because the global ERA5 grid stops at 358.5°.
#
# %%
proj_nps = ccrs.NorthPolarStereo(central_longitude=-100)
proj_pc  = ccrs.PlateCarree()
ev = config.EVENT_DATE

ds = xr.open_dataset(
    ERA5_DIR / f"era5_{ev.year:04d}-{ev.month:02d}-{ev.day:02d}_00Z.nc").squeeze()
z500 = ds["z"].sel(pressure_level=500).values / 9.81 / 10.0   # m -> dam
panel_lats = ds.latitude.values
panel_lons = ds.longitude.values
# Close the 0/360 seam
z500_c, lons_c = add_cyclic_point(z500, coord=panel_lons)

vmin = float(np.nanpercentile(z500_c, 1))
vmax = float(np.nanpercentile(z500_c, 99))
print(f"Event Z500 range: [{vmin:.0f}, {vmax:.0f}] dam")

fig = plt.figure(figsize=(7.5, 7))
ax = fig.add_subplot(1, 1, 1, projection=proj_nps)
ax.set_extent([-180, 180, 20, 90], crs=proj_pc)
ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4")
ax.add_feature(cfeature.BORDERS, lw=0.2, edgecolor="0.6")
cf = ax.contourf(lons_c, panel_lats, z500_c,
                 levels=np.linspace(vmin, vmax, 21),
                 cmap="RdYlBu_r", transform=proj_pc, extend="both")
cs = ax.contour(lons_c, panel_lats, z500_c,
                levels=np.arange(int(vmin//8*8), int(vmax)+8, 8),
                colors="black", linewidths=0.4, transform=proj_pc)
ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f")
# Case domain box from config
ax.plot([config.LON_W, config.LON_E, config.LON_E, config.LON_W, config.LON_W],
        [config.LAT_S, config.LAT_S, config.LAT_N, config.LAT_N, config.LAT_S],
        color="red", lw=1.5, ls="--", transform=proj_pc)
fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7,
             label="Z500 [dam]")
ax.set_title(f"ERA5 Z500 — Event {config.EVENT_DATE} 00Z (red = case domain)",
             fontsize=12, fontweight="bold")
fig.savefig(STEP_DIR/"z500_evolution_5panel.png", dpi=150, bbox_inches="tight")
print("✓ Saved: z500_evolution_5panel.png")
plt.show()

# %% [markdown]
# ## 5. Summary
#
# | Item | Value |
# |------|-------|
# | Files | 1 (event hour only) |
# | Levels | 8 (1000–200 hPa) |
# | Variables | t, u, v, z |
# | Grid | NH 0–90°N × 0–358.5°E, 1.5° |
# | Event | config.EVENT_DATE 00Z |
# | Climatology | pre-computed 30-yr (no download here) |
#
# **→ Step 03**: 30-year climatology + event snapshot subset to the case domain.
#
