#!/usr/bin/env python3
"""Wu Python — Step 08: Visualize PV fields from Pass A/B (pure Python)."""
import numpy as np, matplotlib.pyplot as plt, cartopy.crs as ccrs, cartopy.feature as cfeature
from pathlib import Path; import sys
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path: sys.path.insert(0, str(_root))
from wu_python.core.grid import LON2D, LAT2D
STEP_DIR = Path(__file__).resolve().parent
PY_OUT = Path(_root) / "data" / "wu_python_out"
print("=" * 60)
print("Wu Python — Step 08: PV visualization (pure Python)")
print("=" * 60)
ab = np.load(PY_OUT / "pass_ab.npz")
Q_event, Q_mean = ab["Q_event"], ab["Q_mean"]
Q_anom = Q_event - Q_mean
k250_Q = 6
proj = ccrs.LambertConformal(central_longitude=-105, central_latitude=50); pc = ccrs.PlateCarree()
fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": proj})
for ax, (data, title) in zip(axes, [(Q_mean[k250_Q], "Mean PV"), (Q_anom[k250_Q], "PV Anomaly")]):
    q = np.where(data >= 9999., np.nan, data); vm = np.nanpercentile(np.abs(q), 98)
    ax.set_extent([-175, -35, 5, 88], crs=pc); ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4")
    cf = ax.pcolormesh(LON2D, LAT2D, q, cmap="RdBu_r", transform=pc, vmin=-vm, vmax=vm)
    plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02); ax.set_title(title, fontsize=10)
plt.suptitle("Wu Python — Mean PV & PV Anomaly @ 250 hPa", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(STEP_DIR/"wu_vs_era5_pv_250hpa.png", dpi=150, bbox_inches="tight"); plt.close()
print("  ✓ Saved: wu_vs_era5_pv_250hpa.png")
print("\n✓ Step 08 complete → Step 09: PV advection")
