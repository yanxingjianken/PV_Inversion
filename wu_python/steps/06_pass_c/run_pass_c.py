#!/usr/bin/env python3
"""Wu Python — Step 06: Pass C — Total balanced PV inversion.

Reads the Pass C output (event_bal.out) from the Wu Fortran pipeline and
generates diagnostic plots. Pure-Python BALNC implementation is pending;
this step bridges through existing Fortran outputs for now.

Generates 2 diagnostic plots:
  - pass_c_psi_refinement.png  — Pass B ψ vs Pass C ψ refinement @ 500 hPa
  - pass_c_psi_all_levels.png   — ψ at all 10 σ-levels
"""
import numpy as np, matplotlib.pyplot as plt, cartopy.crs as ccrs, cartopy.feature as cfeature
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path: sys.path.insert(0, str(_root))
from wu_python.core.grid import NY, NX, LON2D, LAT2D

STEP_DIR = Path(__file__).resolve().parent
WU_DIR = Path(_root) / "data" / "wu_in"
PY_OUT = Path(_root) / "data" / "wu_python_out"; PY_OUT.mkdir(parents=True, exist_ok=True)
NW, block = 10, NY * NX

def read_wu_ascii(fp):
    data = []
    with open(fp) as f:
        for line in f:
            for tok in line.split(): data.append(float(tok))
    return np.array(data[:8]), np.array(data[8:])

print("=" * 60)
print("Wu Python — Step 06: Pass C — Total Balanced PV Inversion")
print("  Reading Fortran event_bal.out output")
print("  ⚠ Pure-Python BALNC pending (wu_python/core/balance.py)")
print("=" * 60)

bal_path = WU_DIR / "event_bal.out"
if not bal_path.exists():
    raise FileNotFoundError(f"{bal_path} not found — run wu/steps/06 first")

_, vals = read_wu_ascii(bal_path)
n_expected = 2 * NW * block
if len(vals) != n_expected:
    raise ValueError(f"event_bal.out has {len(vals)} values, expected {n_expected}")
print(f"  Values: {len(vals)} (expected {n_expected})")

H_bal = np.stack([vals[k*block:(k+1)*block].reshape(NY, NX) for k in range(NW)], axis=0)
PSI_bal = np.stack([vals[(NW+k)*block:(NW+1+k)*block].reshape(NY, NX) for k in range(NW)], axis=0)

# Read Pass B ψ for comparison
_, vals_b = read_wu_ascii(WU_DIR / "event_h.out")
PSI_b = np.stack([vals_b[(NW+k)*block:(NW+1+k)*block].reshape(NY, NX) for k in range(NW)], axis=0)

np.savez(PY_OUT / "pass_c.npz", H_bal=H_bal, PSI_bal=PSI_bal, PSI_pass_b=PSI_b)
print("  ✓ Saved: pass_c.npz")

# ---- Plot 1: ψ Refinement @ 500 hPa ----
proj = ccrs.LambertConformal(central_longitude=-105, central_latitude=50); pc = ccrs.PlateCarree()
k500 = 5
fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": proj})
titles = ["Pass B ψ", "Pass C ψ (balanced)", "Δψ (C − B)"]
fields = [PSI_b[k500]*1e-5, PSI_bal[k500]*1e-5, (PSI_bal[k500]-PSI_b[k500])*1e-5]
for ax, title, field in zip(axes, titles, fields):
    vm = np.nanpercentile(np.abs(field), 98) if title.startswith("Δ") else np.nanpercentile(np.abs(field), 98)
    ax.set_extent([-175, -35, 5, 88], crs=pc); ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4")
    cf = ax.pcolormesh(LON2D, LAT2D, field, cmap="RdBu_r", transform=pc, vmin=-vm, vmax=vm)
    plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02, label="ψ × 10⁵"); ax.set_title(title, fontsize=10)
plt.suptitle("Wu Python — Pass C: ψ Refinement @ 500 hPa", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(STEP_DIR/"pass_c_psi_refinement.png", dpi=150, bbox_inches="tight"); plt.close()
print("  ✓ Saved: pass_c_psi_refinement.png")

# ---- Plot 2: ψ @ all levels ----
fig, axes = plt.subplots(2, 5, figsize=(24, 10), subplot_kw={"projection": proj})
axes = axes.ravel()
pr = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]
for k in range(NW):
    ax = axes[k]; field = PSI_bal[k] * 1e-5
    vm = np.nanpercentile(np.abs(field), 98) if np.nanmax(np.abs(field)) > 0 else 1
    ax.set_extent([-175, -35, 5, 88], crs=pc); ax.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor="0.5")
    cf = ax.pcolormesh(LON2D, LAT2D, field, cmap="RdBu_r", transform=pc, vmin=-vm, vmax=vm)
    plt.colorbar(cf, ax=ax, shrink=0.6, pad=0.02); ax.set_title(f"{pr[k]} hPa", fontsize=9)
plt.suptitle("Wu Python — Pass C: Balanced ψ — All Levels", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig(STEP_DIR/"pass_c_psi_all_levels.png", dpi=150, bbox_inches="tight"); plt.close()
print("  ✓ Saved: pass_c_psi_all_levels.png")
print("\n✓ Step 06 complete — 2 plots → Step 07: Pass D piecewise inversion")
