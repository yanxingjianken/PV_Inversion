#!/usr/bin/env python3
"""Wu Python — Step 05: Pass A (mean PV+ψ) + Pass B (event PV+ψ).

Pure-Python Ertel PV + balanced streamfunction computation.
Runs Pass A and Pass B in parallel via ProcessPoolExecutor.

Generates 3 diagnostic plots mirroring wu/steps/05_wu_pass_ab/:
  - wu_mean_pv_3levels.png    — Mean PV @ 500/300/250 hPa (Wu internal units)
  - wu_pv_anomaly_3levels.png — PV anomaly Q_event − Q_mean
  - wu_mean_psi_500hpa.png     — Balanced ψ @ 500 hPa
"""
import numpy as np, matplotlib.pyplot as plt, cartopy.crs as ccrs, cartopy.feature as cfeature
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path: sys.path.insert(0, str(_root))
import config as root_config
from wu_python.core.io import read_wu_grid
from wu_python.core.grid import NY, NX, LON2D, LAT2D
from wu_python.core.pv_calc import compute_relative_vorticity, compute_ertel_pv_wu, invert_vorticity_balanced
from wu_python.config import N_WORKERS

STEP_DIR, NW, NW_PV = Path(__file__).resolve().parent, 10, 8
WU_IN = Path(root_config.WU_IN_DIR)
PY_OUT = Path(root_config.DATA_DIR) / "wu_python_out"; PY_OUT.mkdir(parents=True, exist_ok=True)

def _process(name, gf):
    print(f"  [{name}] Reading {gf.name} …")
    H, TH, U, V = read_wu_grid(gf)
    VOR = compute_relative_vorticity(U, V)
    Q = compute_ertel_pv_wu(U, V, TH)
    PSI = invert_vorticity_balanced(VOR, H, omega=1.75, max_iter=300)
    print(f"  [{name}] Done — ψ range [{PSI.min():.0f}, {PSI.max():.0f}]")
    return {"name": name, "H": H, "TH": TH, "U": U, "V": V, "VOR": VOR, "Q": Q, "PSI": PSI}

print("=" * 60)
print(f"Wu Python — Step 05: Pass A + Pass B  (grid: {NY}×{NX}×{NW})")
print("=" * 60)

tasks = [("mean", WU_IN / "mean.grid"), ("event", WU_IN / "event.grid")]
if N_WORKERS >= 2:
    with ProcessPoolExecutor(max_workers=2) as ex:
        futures = {ex.submit(_process, n, gf): n for n, gf in tasks}
        results = {}
        for f in futures:
            r = f.result()
            results[futures[f]] = r
else:
    results = {n: _process(n, gf) for n, gf in tasks}
mean, event = results["mean"], results["event"]

np.savez(PY_OUT / "pass_ab.npz", Q_mean=mean["Q"], PSI_mean=mean["PSI"], VOR_mean=mean["VOR"],
         H_mean=mean["H"], Q_event=event["Q"], PSI_event=event["PSI"], VOR_event=event["VOR"],
         H_event=event["H"])
print("  ✓ Saved: pass_ab.npz")

# ---- Plot 1: Mean PV @ 3 levels ----
proj = ccrs.LambertConformal(central_longitude=-105, central_latitude=50); pc = ccrs.PlateCarree()
lvls = {2: "500 hPa", 5: "300 hPa", 7: "250 hPa"}
fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": proj})
for ax, (k, lbl) in zip(axes, lvls.items()):
    q = np.where(mean["Q"][k] >= 9999., np.nan, mean["Q"][k]); vm = np.nanpercentile(np.abs(q), 98)
    ax.set_extent([-175, -35, 5, 88], crs=pc); ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4")
    cf = ax.pcolormesh(LON2D, LAT2D, q, cmap="RdBu_r", transform=pc, vmin=-vm, vmax=vm)
    plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02, label="Wu Q (internal)"); ax.set_title(f"Mean PV @ {lbl}", fontsize=10)
plt.suptitle("Wu Python — Pass A: Mean Ertel PV (internal units)", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(STEP_DIR / "wu_mean_pv_3levels.png", dpi=150, bbox_inches="tight"); plt.close()
print("  ✓ Saved: wu_mean_pv_3levels.png")

# ---- Plot 2: PV Anomaly ----
mask = (event["Q"] >= 9999.) | (mean["Q"] >= 9999.); Q_anom = np.where(mask, np.nan, event["Q"] - mean["Q"])
fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": proj})
for ax, (k, lbl) in zip(axes, lvls.items()):
    vm = np.nanpercentile(np.abs(Q_anom[k]), 95)
    ax.set_extent([-175, -35, 5, 88], crs=pc); ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4")
    cf = ax.pcolormesh(LON2D, LAT2D, Q_anom[k], cmap="RdBu_r", transform=pc, vmin=-vm, vmax=vm)
    plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02, label="Wu Q anomaly"); ax.set_title(f"Q′ @ {lbl}", fontsize=10)
plt.suptitle("Wu Python — Pass A+B: PV Perturbation", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(STEP_DIR / "wu_pv_anomaly_3levels.png", dpi=150, bbox_inches="tight"); plt.close()
print("  ✓ Saved: wu_pv_anomaly_3levels.png")

# ---- Plot 3: Mean ψ @ 500 hPa ----
k500 = 5; fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={"projection": proj})
ax.set_extent([-175, -35, 5, 88], crs=pc)
ax.add_feature(cfeature.COASTLINE, lw=0.5, edgecolor="0.3"); ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.2, edgecolor="0.5")
psi = mean["PSI"][k500] * 1e-5; vm = np.nanpercentile(np.abs(psi), 98)
cf = ax.pcolormesh(LON2D, LAT2D, psi, cmap="RdBu_r", transform=pc, vmin=-vm, vmax=vm)
plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02, label="ψ × 10⁵ [m²/s]")
ax.set_title(f"Wu Python — Pass A: ψ @ 500 hPa  [{psi.min():.0f}, {psi.max():.0f}]", fontsize=11, fontweight="bold")
plt.tight_layout(); plt.savefig(STEP_DIR / "wu_mean_psi_500hpa.png", dpi=150, bbox_inches="tight"); plt.close()
print("  ✓ Saved: wu_mean_psi_500hpa.png")
print("\n✓ Step 05 complete — 3 plots → Step 06: Pass C total balance")
