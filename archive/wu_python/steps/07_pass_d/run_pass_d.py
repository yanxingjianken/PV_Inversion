#!/usr/bin/env python3
"""Wu Python — Step 07: Pass D — Piecewise perturbation PV inversion.

Reads event_pert.out from Wu Fortran pipeline. Pure-Python BALP pending.

Generates:
  - pass_d_psi_3pieces_250hpa.png
  - pass_d_piece3_noise.png
"""
import numpy as np, matplotlib.pyplot as plt, cartopy.crs as ccrs, cartopy.feature as cfeature
from pathlib import Path
from scipy.ndimage import gaussian_filter
import sys
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path: sys.path.insert(0, str(_root))
from wu_python.core.grid import NY, NX, LON2D, LAT2D
STEP_DIR = Path(__file__).resolve().parent
WU_DIR = Path(_root) / "data" / "wu_in"
PY_OUT = Path(_root) / "data" / "wu_python_out"; PY_OUT.mkdir(parents=True, exist_ok=True)
NW, block, NPIECES = 10, NY * NX, 3
def read_wu_ascii(fp):
    data = []
    with open(fp) as f:
        for line in f:
            for tok in line.split(): data.append(float(tok))
    return np.array(data[:8]), np.array(data[8:])
print("=" * 60)
print(f"Wu Python — Step 07: Pass D — {NPIECES}-piece perturbation")
print("  Reading Fortran event_pert.out  (pure-Python BALP pending)")
print("=" * 60)
pert_path = WU_DIR / "event_pert.out"
if not pert_path.exists():
    raise FileNotFoundError(f"{pert_path} missing — run wu/steps/07 first")
_, vals = read_wu_ascii(pert_path)
n_per = 2 * NW * block
if len(vals) != NPIECES * n_per:
    raise ValueError(f"Got {len(vals)} values, expected {NPIECES * n_per}")
PSI_p = np.zeros((NPIECES, NW, NY, NX)); H_p = np.zeros((NPIECES, NW, NY, NX))
for ip in range(NPIECES):
    off = ip * n_per
    for k in range(NW):
        H_p[ip,k] = vals[off+k*block:(off+(k+1)*block)].reshape(NY,NX)
        PSI_p[ip,k] = vals[off+(NW+k)*block:(off+(NW+1+k)*block)].reshape(NY,NX)
np.savez(PY_OUT/"pass_d.npz", PSI_pieces=PSI_p, H_pieces=H_p)
print("  ✓ Saved: pass_d.npz")
# Plots
proj = ccrs.LambertConformal(central_longitude=-105, central_latitude=50); pc = ccrs.PlateCarree()
k250 = 7
fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": proj})
for ip in range(3):
    ax = axes[ip]; fld = PSI_p[ip,k250] * 1e-5
    if ip == 2: fld = gaussian_filter(np.nan_to_num(fld), sigma=1.5)
    vm = max(np.nanpercentile(np.abs(fld), 98), 0.01)
    ax.set_extent([-175, -35, 5, 88], crs=pc); ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4")
    cf = ax.pcolormesh(LON2D, LAT2D, fld, cmap="RdBu_r", transform=pc, vmin=-vm, vmax=vm)
    plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02, label="ψ′ × 10⁵")
    ax.set_title(["(a) Lower", "(b) Middle", "(c) Upper [smoothed]"][ip], fontsize=10)
plt.suptitle("Wu Python — Pass D: ψ′ @ 250 hPa", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(STEP_DIR/"pass_d_psi_3pieces_250hpa.png", dpi=150, bbox_inches="tight"); plt.close()
print("  ✓ pass_d_psi_3pieces_250hpa.png")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": proj})
for ax, fld, t in zip(axes, [PSI_p[2,k250]*1e-5, gaussian_filter(np.nan_to_num(PSI_p[2,k250]*1e-5), sigma=1.5)], ["Raw", "σ=1.5"]):
    vm = max(np.nanpercentile(np.abs(fld), 98), 0.01)
    ax.set_extent([-175, -35, 5, 88], crs=pc); ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4")
    cf = ax.pcolormesh(LON2D, LAT2D, fld, cmap="RdBu_r", transform=pc, vmin=-vm, vmax=vm)
    plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02); ax.set_title(t, fontsize=10)
plt.suptitle("Wu Python — Pass D: Piece 3 Noise", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(STEP_DIR/"pass_d_piece3_noise.png", dpi=150, bbox_inches="tight"); plt.close()
print("  ✓ pass_d_piece3_noise.png")
print("✓ Step 07 complete → Step 08")
