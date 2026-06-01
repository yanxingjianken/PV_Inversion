# %% [markdown]
# Step 01 — Download ERA5 for Climatology Window & Polar Map Viz
#
# Downloads all daily ERA5 snapshots needed for the running-mean climatology,
# computed from `config.CLIM_WINDOW_DAYS` (symmetric around the event date).
#
# **Output**: `data/era5/era5_YYYY-MM-DD_00Z.nc` (CLIM_WINDOW_DAYS files)

# %%
import cdsapi, os, numpy as np, xarray as xr
import matplotlib.pyplot as plt, cartopy.crs as ccrs, cartopy.feature as cfeature
from pathlib import Path

import sys; from pathlib import Path as _Path
_sys_path_root = str(_Path(__file__).resolve().parent.parent.parent)
if _sys_path_root not in sys.path: sys.path.insert(0, _sys_path_root)
import config
STEP_DIR = _Path(__file__).resolve().parent
ERA5_DIR = _Path(config.ERA5_DIR); ERA5_DIR.mkdir(parents=True, exist_ok=True)

LEVELS = ["1000","925","850","700","600","500","400","300","250","200"]
VARS   = ["temperature","u_component_of_wind","v_component_of_wind","geopotential"]

# Derive date list from config
DATES = [(d.year, d.month, d.day) for d in config.CLIM_DATES]
print(f"Downloading {len(DATES)} days ({config.CLIM_WINDOW_DAYS}-day window) to {ERA5_DIR}")
print(f"  Range: {config.CLIM_START} to {config.CLIM_END}")
print(f"  Event: {config.EVENT_DATE}")

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
# ## 3. Inspect All Days
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
# ## 4. Polar Map — Z500 Evolution (5 evenly-spaced days through window)
#
# %%
proj_nps = ccrs.NorthPolarStereo(central_longitude=-100)
proj_pc  = ccrs.PlateCarree()
N = len(config.CLIM_DATES)
key_idx = sorted(set([0, N//4, N//2, (3*N)//4, N-1]))
key_dates = [config.CLIM_DATES[i] for i in key_idx]

# Pre-load all panel Z500 fields to derive a shared colour range
panel_z = []
for d in key_dates:
    ds = xr.open_dataset(ERA5_DIR / f"era5_{d.year:04d}-{d.month:02d}-{d.day:02d}_00Z.nc").squeeze()
    panel_z.append(ds["z"].sel(pressure_level=500).values / 9.81 / 10.0)  # m -> dam
panel_lats = ds.latitude.values
panel_lons = ds.longitude.values
vmin = float(np.nanpercentile(np.stack(panel_z), 1))
vmax = float(np.nanpercentile(np.stack(panel_z), 99))
print(f"Z500 range across window: [{vmin:.0f}, {vmax:.0f}] dam")

fig, axes = plt.subplots(1, len(key_dates), figsize=(4.4*len(key_dates), 5),
                         subplot_kw={"projection": proj_nps})
cf = None
for ax, d, z500 in zip(np.atleast_1d(axes), key_dates, panel_z):
    ax.set_extent([-180,180,20,90], crs=proj_pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4")
    ax.add_feature(cfeature.BORDERS, lw=0.2, edgecolor="0.6")
    cf = ax.contourf(panel_lons, panel_lats, z500,
                     levels=np.linspace(vmin, vmax, 21),
                     cmap="RdYlBu_r", transform=proj_pc, extend="both")
    cs = ax.contour(panel_lons, panel_lats, z500,
                    levels=np.arange(int(vmin//8*8), int(vmax)+8, 8),
                    colors="black", linewidths=0.4, transform=proj_pc)
    ax.clabel(cs, inline=True, fontsize=5, fmt="%.0f")
    # Case domain box from config
    ax.plot([config.LON_W, config.LON_E, config.LON_E, config.LON_W, config.LON_W],
            [config.LAT_S, config.LAT_S, config.LAT_N, config.LAT_N, config.LAT_S],
            color="red", lw=1.5, ls="--", transform=proj_pc)
    is_event = (d == config.EVENT_DATE)
    ax.set_title(f"{d.strftime('%b %d')}{' ← EVENT' if is_event else ''}",
                 fontsize=10, fontweight="bold" if is_event else "normal",
                 color="red" if is_event else "black")

fig.subplots_adjust(bottom=0.18, top=0.88, wspace=0.05)
cax = fig.add_axes([0.25, 0.10, 0.50, 0.025])
fig.colorbar(cf, cax=cax, orientation="horizontal", label="Z500 [dam]")
fig.suptitle(f"ERA5 Z500 — {config.CLIM_WINDOW_DAYS}-day window centered on "
             f"{config.EVENT_DATE} (red = case domain)",
             fontsize=13, fontweight="bold")
fig.savefig(STEP_DIR/"z500_evolution_5panel.png", dpi=150, bbox_inches="tight")
print("✓ Saved: z500_evolution_5panel.png")
plt.show()

# %% [markdown]
# ## 5. Summary
#
# | Item | Value |
# |------|-------|
# | Days | 11 (Jan 3–13 2025 00Z) |
# | Levels | 10 native (1000–200 hPa) |
# | Variables | t, u, v, z |
# | Grid | 61 lat × 240 lon, 1.5° |
# | NaN | 0 across all days |
# | Event | Jan 8 (center of 11-day window) |
#
# **→ Step 02**: Grid & σ-coordinate explanation + CA domain subset.
#
