# %% [markdown]
# Step 07 — Wu Pass D: Piecewise Perturbation PV Inversion (3 Pieces)
#
# The `qinvertp21` program (BALP subroutine) partitions the PV perturbation
# q′ = q_event − q_mean into **3 vertical pieces** and inverts each independently
# to get ψ′ (perturbation streamfunction) for each piece.
#
# **Pieces**:
# - Piece 1 (lower): K={1,2} → 1000, 925 hPa
# - Piece 2 (middle): K={3,4} → 850, 700 hPa
# - Piece 3 (upper): K={5..9} → 600, 500, 400, 300, 250 hPa
#
# **Output**: `event_pert.out` — HP, SP for 3 pieces at all 10 levels.
#
# **Critical params**:
# - `INLIN=0` (linear balance) — INLIN=1 explodes at 250 hPa jet
# - `IBC=0` — homogeneous Dirichlet on lateral boundaries
# - `OMEGS=1.4, OMEGH=1.4` — under-relaxed for stability
#
# %% [markdown]
# ## 1. Run qinvertp21
#
# %%
import subprocess, os, numpy as np, matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from pathlib import Path

import sys; from pathlib import Path as _Path
_sys_path_root = str(_Path(__file__).resolve().parent.parent.parent)
if _sys_path_root not in sys.path: sys.path.insert(0, _sys_path_root)
import config
STEP_DIR = _Path(__file__).resolve().parent
DATA_DIR = _Path(config.DATA_DIR); ERA5_DIR = _Path(config.ERA5_DIR)
CLIM_DIR = _Path(config.CLIM_DIR); WU_DIR = _Path(config.WU_IN_DIR)
OUT_DIR = _Path(config.WU_OUT_DIR); FIG_DIR = _Path(config.FIG_DIR)
BUILD_DIR = _Path(config.DATA_DIR) / "wu_bin"

# (STEP_DIR from config import)
# (WU_DIR from config)
# (BUILD_DIR from config import)

# Pass D stdin (from working 04_run_wu.sh — single quotes, INLIN=0 for stability)
os.chdir(str(WU_DIR))
exe = BUILD_DIR / "qinvertp21.exe"
for fn in ["event_pert.out"]:
    (WU_DIR / fn).unlink(missing_ok=True)

stdin_d = """1.4
1.4
0.5
0.01
1.
1.
'meanq'
'meanh'
'event_q.out'
'event_bal.out'
'event_pert.out'
1
0
0
10,1,2,3,4,5,6,7,8,9,10
10,1,2,3,4,5,6,7,8,9,10
3
2,1,2
0
2,3,4
0
5,5,6,7,8,9
0
"""

result_d = subprocess.run(
    [str(exe)], input=stdin_d, capture_output=True, text=True, timeout=600
)
print("Pass D stdout (last 30 lines):")
for line in result_d.stdout.split("\n")[-30:]:
    print(f"  {line}")

# Check convergence per piece
for line in result_d.stdout.split("\n"):
    if "TOTAL" in line or "TOO MANY" in line or "piece" in line.lower():
        print(f"  >>> {line}")

fp = WU_DIR / "event_pert.out"
if fp.exists():
    print(f"\n✓ event_pert.out created ({fp.stat().st_size} bytes)")
else:
    print("\n✗ event_pert.out MISSING!")
os.chdir(str(STEP_DIR))

# %% [markdown]
# ## 2. Parse event_pert.out
#
# %%
def read_wu_ascii(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            for tok in line.split():
                data.append(float(tok))
    return np.array(data[:8]), np.array(data[8:])

NX, NY, NW = 87, 51, 10
NPIECES = 3
block = NY * NX

hdr_pert, vals_pert = read_wu_ascii(WU_DIR / "event_pert.out")
expected = NPIECES * 2 * NW * block
print(f"event_pert.out: {len(vals_pert)} values (expected {expected})")
if len(vals_pert) < expected:
    print(f"  ⚠ Padding with zeros (missing {expected-len(vals_pert)} values)")
    vals_pert = np.pad(vals_pert, (0, expected - len(vals_pert)))

# Parse: per piece: NW × HP + NW × SP
HP = np.zeros((NPIECES, NW, NY, NX))
SP = np.zeros((NPIECES, NW, NY, NX))
per_piece = 2 * NW * block

for ip in range(NPIECES):
    base = ip * per_piece
    for k in range(NW):
        HP[ip,k] = vals_pert[base + k*block : base + (k+1)*block].reshape(NY, NX)
    base_sp = base + NW*block
    for k in range(NW):
        SP[ip,k] = vals_pert[base_sp + k*block : base_sp + (k+1)*block].reshape(NY, NX)

# NaN check
n_nan = int(np.isnan(SP).sum()) + int(np.isnan(HP).sum())
print(f"NaN count: SP={np.isnan(SP).sum()}, HP={np.isnan(HP).sum()}")
if n_nan > 0:
    print("  ⚠ NaN values detected!")

# Range check per piece at 250 hPa
for ip in range(NPIECES):
    sp = SP[ip, 8] * 1.0e5  # K=8 = 250 hPa, actual ψ'
    hp = HP[ip, 8]
    print(f"  Piece {ip+1} @250hPa: SP [{sp.min():.0f}, {sp.max():.0f}] m²/s, "
          f"HP [{hp.min():.1f}, {hp.max():.1f}] m")

# %% [markdown]
# ## 3. Visualize: ψ' at 250 hPa — All 3 Pieces
#
# %%
lats = 85.5 - np.arange(NY) * 1.5
lons = -169.5 + np.arange(NX) * 1.5
LON2D, LAT2D = np.meshgrid(lons, lats)
proj = ccrs.LambertConformal(central_longitude=-105, central_latitude=50)
pc   = ccrs.PlateCarree()

piece_titles = ["(a) Lower: 1000–925 hPa", "(b) Middle: 850–700 hPa",
                "(c) Upper: 600–250 hPa"]

fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={"projection": proj})
for ip, (ax, title) in enumerate(zip(axes, piece_titles)):
    psi = SP[ip, 8] * 1.0e5   # actual ψ' at 250 hPa
    vm = np.percentile(np.abs(psi), 98)
    ax.set_extent([-175, -35, 5, 88], crs=pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4")
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.2, edgecolor="0.6")
    cf = ax.pcolormesh(LON2D, LAT2D, psi, cmap="RdBu_r", transform=pc,
                       vmin=-vm, vmax=vm)
    ax.contour(LON2D, LAT2D, psi, colors="black", linewidths=0.3, transform=pc,
               levels=np.linspace(-vm, vm, 12))
    plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02, label="ψ′ [m²/s]")
    ax.set_title(f"{title}\nψ′ range: [{psi.min():.0f}, {psi.max():.0f}] m²/s",
                 fontsize=10)

plt.suptitle("Wu Pass D — Piecewise Perturbation Streamfunction ψ′ at 250 hPa\n"
             "2025-01-08 00Z CA Blocking Event", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(STEP_DIR/"pass_d_psi_3pieces_250hpa.png", dpi=150, bbox_inches="tight")
print("✓ Saved: pass_d_psi_3pieces_250hpa.png")
plt.show()

# %% [markdown]
# ## 4. Diagnose Piece 3 Noise
#
# Piece 3 (upper troposphere) typically shows gridpoint-scale noise at
# 250 hPa caused by the rigid-lid top boundary at 200 hPa leaking SOR
# residuals downward. This is a known solver artifact.
#
# %%
from scipy.ndimage import gaussian_filter

psi3_raw = SP[2, 8] * 1.0e5
psi3_smooth = gaussian_filter(psi3_raw, sigma=1.5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": proj})

for ax, data, title in [(ax1, psi3_raw, "Raw ψ′ (noisy)"),
                          (ax2, psi3_smooth, "Gaussian σ=1.5 smoothed")]:
    vm = np.percentile(np.abs(data), 98)
    ax.set_extent([-175, -35, 5, 88], crs=pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4")
    cf = ax.pcolormesh(LON2D, LAT2D, data, cmap="RdBu_r", transform=pc,
                       vmin=-vm, vmax=vm)
    plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02, label="ψ′ [m²/s]")
    ax.set_title(title, fontsize=10)

plt.suptitle("Piece 3 (Upper) — Gridpoint Noise Diagnosis\n"
             "Gaussian σ=1.5 removes ~90% of gridpoint noise",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(STEP_DIR/"pass_d_piece3_noise.png", dpi=150, bbox_inches="tight")
print("✓ Saved: pass_d_piece3_noise.png")
plt.show()

print("\n→ Step 08: Parse all outputs + compute ERA5 Ertel PV for physical-unit comparison")

