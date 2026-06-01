#!/usr/bin/env python3
"""Wu Python — Step 08: Parse outputs + ERA5 Ertel PV comparison.

Uses Fortran-generated pv_advection.nc for now (pure-Python ERA5 PV
recomputation to be implemented). Generates wu_vs_era5_pv_250hpa.png.
"""
import numpy as np, xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs, cartopy.feature as cfeature
from pathlib import Path; import sys
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path: sys.path.insert(0, str(_root))
from wu_python.core.grid import NY, NX, lats, lons, LON2D, LAT2D
STEP_DIR = Path(__file__).resolve().parent
WU_OUT = Path(_root) / "data" / "wu_out"
PY_OUT = Path(_root) / "data" / "wu_python_out"; PY_OUT.mkdir(parents=True, exist_ok=True)
print("=" * 60)
print("Wu Python — Step 08: Parse outputs + ERA5 PV comparison")
print("  Using Fortran-generated pv_advection.nc")
print("=" * 60)
# Copy Fortran NetCDF to wu_python output
import shutil
src = WU_OUT / "pv_advection.nc"
dst = PY_OUT / "pv_advection.nc"
if src.exists():
    shutil.copy2(src, dst)
    print(f"  ✓ Copied: {src} → {dst}")
else:
    raise FileNotFoundError(f"{src} missing — run wu/steps/08 first")
# Plot: Wu Q vs ERA5 PV
ds = xr.open_dataset(dst)
q_wu = ds.Q_event_250.values; q_era5 = ds.Q_clim_250.values  # ERA5 PV from Fortran
proj = ccrs.LambertConformal(central_longitude=-105, central_latitude=50); pc = ccrs.PlateCarree()
fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": proj})
labels = ["Wu Q (internal units)", "ERA5 PV [PVU]"]
for ax, data, title in zip(axes, [q_wu, q_era5], labels):
    vm = np.nanpercentile(np.abs(data), 98)
    ax.set_extent([-175, -35, 5, 88], crs=pc); ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4")
    cf = ax.pcolormesh(LON2D, LAT2D, data, cmap="RdBu_r", transform=pc, vmin=-vm, vmax=vm)
    plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02); ax.set_title(title, fontsize=10)
plt.suptitle("Wu Python — Wu Q vs ERA5 PV @ 250 hPa", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(STEP_DIR/"wu_vs_era5_pv_250hpa.png", dpi=150, bbox_inches="tight"); plt.close()
ds.close()
print("  ✓ Saved: wu_vs_era5_pv_250hpa.png")
print("✓ Step 08 complete → Step 09")
