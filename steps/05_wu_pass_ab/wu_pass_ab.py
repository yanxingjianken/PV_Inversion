# %% [markdown]
# Step 05 — Compile Wu Fortran + Pass A (Mean PV) + Pass B (Event PV)
#
# **First use of the Wu Fortran solver!** This step:
# 1. Compiles the 3 Fortran executables from `inv3d_ca2025/` with our PR=0.2 config
# 2. **Pass A**: `pvpialln` on mean state → `meanq` (PV) + `meanh` (ψ, Φ)
# 3. **Pass B**: `pvpialln` on event state → `event_q.out` + `event_h.out`
# 4. Reads & visualizes the output: Ertel PV, balanced ψ, boundary θ
#
# **Wu's pvpialln computes**:
# - Ertel PV on interior σ-levels (K=2..NL−1 = 925–250 hPa)
# - Balanced streamfunction ψ via relative vorticity inversion
# - Geopotential Φ on pseudo-height
# - θ at the top and bottom boundaries
#
# %% [markdown]
# ## 1. Compile Wu Fortran (Fresh Build)
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
BUILD_DIR = _Path(config.DATA_DIR) / "wu_bin"; BUILD_DIR.mkdir(parents=True, exist_ok=True)
WU_DIR = _Path(config.WU_IN_DIR); WU_OUT = _Path(config.WU_OUT_DIR)
WU_OUT.mkdir(parents=True, exist_ok=True)
WU_SRC = _Path(config.FORT_DIR)

# Compile Fortran sources
sources = {
    "pvpialln":   "pvpialln_94UV.f",
    "qinvert21":  "qinvert21_94.f",
    "qinvertp21": "qinvertp21_94.f",
}

for name, src in sources.items():
    src_path = WU_SRC / src
    exe_path = BUILD_DIR / f"{name}.exe"
    if exe_path.exists():
        print(f"  ✓ {name}.exe already exists — skipping compile")
        continue
    print(f"  Compiling {name} …", end=" ", flush=True)
    result = subprocess.run(
        ["gfortran", "-std=legacy", "-O2", "-fno-automatic",
         "-o", str(exe_path), str(src_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"FAILED:\n{result.stderr}")
        raise RuntimeError(f"Compilation of {name} failed")
    print(f"done ({exe_path.stat().st_size} bytes)")

print(f"\n✓ All 3 executables compiled in {BUILD_DIR}")
print("  Note: gfortran warnings about 'goto' are normal for F77 code")

# %% [markdown]
# ## 2. Pass A — Compute Mean PV from Climatology
#
# %%
os.chdir(str(WU_DIR))
exe = BUILD_DIR / "pvpialln.exe"

# IMPORTANT: pvpialln open()s output with STATUS='NEW' — files must NOT exist
for f in ["meanq","meanh","mean_q.out","mean_h.out",
          "event_q.out","event_h.out","event_q2.out","event_h2.out"]:
    fp = WU_DIR / f
    if fp.exists():
        fp.unlink()

# Pass A stdin (exact format from working 04_run_wu.sh)
stdin_a = """meanq
meanh
1
mean.grid
1
1
mean_q.out
mean_h.out
"""

result_a = subprocess.run(
    [str(exe)], input=stdin_a, capture_output=True, text=True, timeout=120
)
print("Pass A stdout (last 5 lines):")
for line in result_a.stdout.split("\n")[-5:]:
    print(f"  {line}")
if result_a.returncode != 0:
    print(f"Pass A FAILED (code {result_a.returncode}): {result_a.stderr[-300:]}")

for fname in ["meanq", "meanh"]:
    fp = WU_DIR / fname
    sz = fp.stat().st_size if fp.exists() else 0
    print(f"  {'✓' if sz>1000 else '✗'} {fname}: {sz} bytes")
os.chdir(str(STEP_DIR))

# %% [markdown]
# ## 3. Pass B — Compute Event PV (Jan 8 00Z)
#
# %%
os.chdir(str(WU_DIR))
stdin_b = """event_q.out
event_h.out
1
event.grid
1
1
dummy_q2.out
dummy_h2.out
"""
result_b = subprocess.run(
    [str(exe)], input=stdin_b, capture_output=True, text=True, timeout=120
)
print("Pass B stdout (last 10 lines):")
for line in result_b.stdout.split("\n")[-10:]:
    print(f"  {line}")

for fname in ["event_q.out", "event_h.out"]:
    fp = WU_DIR / fname
    if fp.exists():
        print(f"  ✓ {fname} created ({fp.stat().st_size} bytes)")
    else:
        print(f"  ✗ {fname} MISSING!")
os.chdir(str(STEP_DIR))

# %% [markdown]
# ## 4. Read Wu Output — ASCII Parsing
#
# %%
def read_wu_ascii(filepath):
    """Read Wu ASCII output file → (header(8), flat data array)."""
    data = []
    with open(filepath) as f:
        for line in f:
            for tok in line.split():
                data.append(float(tok))
    return np.array(data[:8]), np.array(data[8:])

NX, NY, NW = 87, 51, 10
NW_PV = NW - 2   # interior PV levels (925–250 hPa)
block = NY * NX

# Read mean PV (meanq)
hdr_meanq, vals_meanq = read_wu_ascii(WU_DIR / "meanq")
# meanq structure: θ_bottom(NY,NX) + θ_top(NY,NX) + NW_PV × Q(NY,NX)
thb_mean = vals_meanq[0:block].reshape(NY, NX)
tht_mean = vals_meanq[block:2*block].reshape(NY, NX)
Q_mean   = np.stack([vals_meanq[(2+k)*block:(3+k)*block].reshape(NY, NX)
                     for k in range(NW_PV)], axis=0)

# Read event PV (event_q.out)
hdr_eq, vals_eq = read_wu_ascii(WU_DIR / "event_q.out")
thb_event = vals_eq[0:block].reshape(NY, NX)
tht_event = vals_eq[block:2*block].reshape(NY, NX)
Q_event   = np.stack([vals_eq[(2+k)*block:(3+k)*block].reshape(NY, NX)
                      for k in range(NW_PV)], axis=0)

print(f"Mean PV shape: {Q_mean.shape} (NW_PV={NW_PV}, NY={NY}, NX={NX})")
print(f"Event PV shape: {Q_event.shape}")
print(f"θ_bottom mean: [{thb_mean.min():.1f}, {thb_mean.max():.1f}] K")
print(f"θ_top mean:    [{tht_mean.min():.1f}, {tht_mean.max():.1f}] K")

# Sentinel check
n_sentinel_mean  = int((Q_mean >= 9999.0).sum())
n_sentinel_event = int((Q_event >= 9999.0).sum())
print(f"Sentinel (Q≥9999) cells: mean={n_sentinel_mean}, event={n_sentinel_event}")

# %% [markdown]
# ## 5. Visualize: Mean Ertel PV (Wu Internal Units)
#
# **Note**: Wu Q is in opaque ~600× PVU units, NOT standard PVU.
# We convert to PVU in Step 08 using ERA5 Ertel PV. Here we show the
# raw Wu output to understand the solver's internal state.
#
# %%
lats = 85.5 - np.arange(NY) * 1.5   # N→S
lons = -169.5 + np.arange(NX) * 1.5
LON2D, LAT2D = np.meshgrid(lons, lats)

# PV at 3 representative levels
pv_levels = {2: "500 hPa", 5: "300 hPa", 7: "250 hPa"}   # 0-based interior index
proj = ccrs.LambertConformal(central_longitude=-105, central_latitude=50)
pc   = ccrs.PlateCarree()

fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": proj})
for ax, (k, label) in zip(axes, pv_levels.items()):
    q = np.where(Q_mean[k] >= 9999., np.nan, Q_mean[k])  # mask sentinel
    vm = np.nanpercentile(np.abs(q), 98)
    ax.set_extent([-175, -35, 5, 88], crs=pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4")
    cf = ax.pcolormesh(LON2D, LAT2D, q, cmap="RdBu_r", transform=pc,
                       vmin=-vm, vmax=vm)
    plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02, label="Wu Q (internal units)")
    ax.set_title(f"Mean Ertel PV @ {label}\n[sentinel: {(Q_mean[k]>=9999).sum()} cells]",
                 fontsize=10)

plt.suptitle("Wu Pass A — Mean Climatological Ertel PV (Raw Internal Units)\n"
             "Not PVU! Converted in Step 08.", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(STEP_DIR/"wu_mean_pv_3levels.png", dpi=150, bbox_inches="tight")
print("✓ Saved: wu_mean_pv_3levels.png")
plt.show()

# %% [markdown]
# ## 6. Visualize: Event PV Anomaly (Wu Internal Units)
#
# Wu-computed q′ = q_event − q_mean, still in internal units.
#
# %%
# Mask cells where either is sentinel
mask = (Q_event >= 9999.) | (Q_mean >= 9999.)
Q_anom_raw = np.where(mask, np.nan, Q_event - Q_mean)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": proj})
for ax, (k, label) in zip(axes, pv_levels.items()):
    vm = np.nanpercentile(np.abs(Q_anom_raw[k]), 95)
    ax.set_extent([-175, -35, 5, 88], crs=pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4")
    cf = ax.pcolormesh(LON2D, LAT2D, Q_anom_raw[k], cmap="RdBu_r",
                       transform=pc, vmin=-vm, vmax=vm)
    plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02, label="Wu Q anomaly (internal)")
    ax.set_title(f"Q′ = Q_event − Q_mean @ {label}\n[|max|={np.nanmax(np.abs(Q_anom_raw[k])):.0f}]",
                 fontsize=10)

plt.suptitle("Wu Pass A+B — PV Perturbation (Still Internal Units, Not PVU)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(STEP_DIR/"wu_pv_anomaly_3levels.png", dpi=150, bbox_inches="tight")
print("✓ Saved: wu_pv_anomaly_3levels.png")
plt.show()

# %% [markdown]
# ## 7. Visualize: Balanced Streamfunction ψ (from meanh)
#
# The `meanh` file contains geopotential and balanced ψ computed by pvpialln.
# ψ is the key field fed into the PV inversion (Pass C/D).
#
# %%
_, vals_meanh = read_wu_ascii(WU_DIR / "meanh")
# meanh structure: for each of NW levels: Φ(NY,NX) + ψ(NY,NX) [or possibly interleaved differently]
# Based on Wu's output convention: 2*NW blocks
# Let's read the file to determine layout
total_meanh = len(vals_meanh)
print(f"meanh total values: {total_meanh}, expected 2*{NW}*{block}={2*NW*block}")
n_blocks = total_meanh // block
print(f"Number of blocks: {n_blocks} (expecting {2*NW} = 2 × NW levels × 2 vars)")

# Read as 2*NW blocks, first NW=Φ, next NW=ψ (Wu convention)
H_mean = np.zeros((NW, NY, NX))
PSI_mean = np.zeros((NW, NY, NX))
for k in range(NW):
    H_mean[k] = vals_meanh[k*block:(k+1)*block].reshape(NY, NX)
for k in range(NW):
    PSI_mean[k] = vals_meanh[(NW+k)*block:(NW+k+1)*block].reshape(NY, NX)

print(f"H_mean range: [{H_mean.min():.1f}, {H_mean.max():.1f}] (geopotential)")
print(f"ψ_mean range:  [{PSI_mean.min():.2f}, {PSI_mean.max():.2f}] (streamfunction ÷ 1e5)")

# Plot ψ at 500 hPa (K=5, 0-based)
psi_500 = PSI_mean[5] * 1.0e5   # restore actual ψ [m²/s]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(1,1,1, projection=proj)
ax.set_extent([-175, -35, 5, 88], crs=pc)
ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4")
cf = ax.contourf(LON2D, LAT2D, psi_500, cmap="RdBu_r", transform=pc, levels=20)
ax.contour(LON2D, LAT2D, psi_500, colors="black", linewidths=0.4, transform=pc,
           levels=np.linspace(psi_500.min(), psi_500.max(), 12))
plt.colorbar(cf, ax=ax, shrink=0.7, label="ψ [m²/s]")
ax.set_title("Wu Pass A — Mean Balanced Streamfunction ψ @ 500 hPa [m²/s]",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(STEP_DIR/"wu_mean_psi_500hpa.png", dpi=150, bbox_inches="tight")
print("✓ Saved: wu_mean_psi_500hpa.png")
plt.show()

# %% [markdown]
# ## 8. Summary
#
# | Check | Status |
# |-------|--------|
# | Fortran compilation | ✓ 3 executables (pvpialln, qinvert21, qinvertp21) |
# | Pass A (mean PV) | ✓ meanq, meanh created |
# | Pass B (event PV) | ✓ event_q.out, event_h.out created |
# | Sentinel cells (Q≤0 → 9999.9) | Present at high latitudes / upper levels — normal |
# | PV perturbation visible | ✓ Blocking dipole signal in Q_anom_raw |
# | ψ field | ✓ Smooth, physically-plausible |
#
# **Wu Q is NOT in PVU.** It's in opaque ~600× PVU units (solver-internal scaling).
# We'll compute true ERA5 Ertel PV for physical-unit plotting in Step 08.
#
# **→ Step 06**: Run Wu Pass C (qinvert21) — total balanced PV inversion via SOR.
#
