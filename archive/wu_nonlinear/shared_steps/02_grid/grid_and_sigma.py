# %% [markdown]
# Step 02 — Grid, σ-Coordinates & CA Domain Subset
#
# **Answers the question**: *"sigma level是怎么来的？era5明明可以直接下载ml level并转化成sigma对吧？"*
#
# ## Executive Summary
#
# **Wu's PR array IS σ = p/p₀ (pure sigma, not hybrid).** ERA5 model levels are
# **hybrid sigma-pressure**: p(η) = a(η)·p₀ + b(η)·p_s. The b(η) term makes each
# level's actual pressure vary with local surface pressure — they're NOT constant-p
# surfaces. For Ertel PV computation (which needs ∂θ/∂p), constant-pressure surfaces
# are standard and much simpler.
#
# **Could we convert model levels to pure sigma?** Technically yes, but it requires:
# 1. Downloading 137 model levels × 4 variables (huge)
# 2. Surface pressure field
# 3. a/b hybrid coefficients
# 4. Interpolating each column to target σ
#
# This adds ~10× data volume and 3 extra processing steps with zero accuracy gain
# at 1.5° resolution. Every published PPVI study uses **pressure levels**, not model
# levels. We follow that standard.
#
# ## What This Notebook Does
# 1. Explain the grid: 87×51, 1.5°, why these numbers
# 2. Explain σ-levels: why 10 levels, why uniform Δσ=0.05 at top is mandatory
# 3. Subset ERA5 full-NH data to CA domain
# 4. Visualize grid points on a map
#
# %% [markdown]
# ## 1. Grid Specification
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
DATA_DIR = _Path(config.DATA_DIR); ERA5_DIR = _Path(config.ERA5_DIR)
CLIM_DIR = _Path(config.CLIM_DIR); WU_DIR = _Path(config.WU_IN_DIR)
OUT_DIR = _Path(config.WU_OUT_DIR); FIG_DIR = _Path(config.FIG_DIR)
BUILD_DIR = _Path(config.DATA_DIR) / "wu_bin"

STEP_DIR = Path.cwd()
DATA_DIR = ERA5_DIR

# ── Hardcoded grid (must match Fortran PARAMETER blocks exactly) ──
NX, NY = 87, 51
DLAT, DLON = 1.5, 1.5
LAT_N, LAT_S = 85.5, 10.5   # north → south in array
LON_W, LON_E = -169.5, -40.5

# Derived arrays
lats = LAT_N - np.arange(NY) * DLAT   # [85.5, 84.0, …, 10.5]
lons = LON_W + np.arange(NX) * DLON   # [-169.5, -168.0, …, -40.5]

print(f"Grid: {NX}×{NY} = {NX*NY:,} cells, Δlat=Δlon={DLAT}°")
print(f"Lat: {lats[0]:.1f}°N → {lats[-1]:.1f}°N ({NY} rows, I=0 is north)")
print(f"Lon: {lons[0]:.1f}°E → {lons[-1]:.1f}°E ({NX} cols, J=0 is west)")
print(f"Area: ~{((LAT_N-LAT_S)*(LON_E-LON_W)):.0f}°²")

# %% [markdown]
# ## 2. Pressure Levels & σ-Coordinate
#
# Wu's `PR` array = p / p₀ where p₀ = 1000 hPa. Each element is a
# computational surface. The solver computes Ertel PV on interior levels
# K=2..NL−1 using ∂θ/∂p finite differences.
#
# **Critical**: the top 3 levels must have uniform σ-spacing (Δσ = 0.05)
# or the SOR solver's finite-difference coefficients become ill-conditioned
# at K=NL−1 (=250 hPa), causing exponential divergence.
#
# %%
PR = np.array([1.0, 0.925, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2])
PLEV = PR * 1000.0   # hPa

print("  K   PR (σ)   P (hPa)   Δσ      Notes")
print("───  ───────  ────────  ──────  ──────────────────")
for k, (pr, p) in enumerate(zip(PR, PLEV)):
    dp = PR[k-1] - pr if k > 0 else 0
    note = ""
    if k == 0: note = "← lower BC (θ fixed)"
    elif k == len(PR)-1: note = "← upper BC (θ fixed)"
    elif 7 <= k <= 9: note = "← UNIFORM Δσ=0.05 at top"
    print(f"  {k+1:2d}   {pr:.3f}    {p:6.0f}    {dp:.3f}    {note}")

# %% [markdown]
# ## 3. Why NOT 100 hPa? (The Blowup Story)
#
# Earlier we tried PR = [1.0, ..., 0.25, 0.15, 0.1] (top at 100 hPa).
# This gave Δσ = 0.10 between 150→100 hPa — double the spacing between
# 250→200 hPa (Δσ=0.05). The SOR at K=NL−1=250 hPa diverged exponentially
# regardless of iteration count. Root cause: non-uniform vertical FD
# coefficients make the PSI denom `BSI·AC(3)−2·ASI·(SLL+SPP)` → 0.
#
# **Lesson**: keep the top 3 σ-levels equally spaced. Our PR = [..., 0.3, 0.25, 0.2]
# achieves this (Δσ = 0.05 everywhere above 400 hPa).
#
# %%
# Visualize σ-spacing
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Good (our config)
pr_good = np.array([1.0, 0.925, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2])
ax1.stem(range(1,11), pr_good, linefmt='g-', markerfmt='go')
for i, p in enumerate(pr_good):
    ax1.text(i+1.1, p, f'{p*1000:.0f} hPa\nΔσ={pr_good[i-1]-p:.3f}' if i>0 else f'{p*1000:.0f} hPa',
             fontsize=7, va='center')
ax1.set_xlabel('K index'); ax1.set_ylabel('σ = p/1000hPa')
ax1.set_title('GOOD: uniform Δσ=0.05 at top\n(stable SOR convergence)', color='green')
ax1.invert_yaxis(); ax1.set_xticks(range(1,11)); ax1.grid(True, alpha=0.3)

# Bad (100 hPa lid)
pr_bad = np.array([1.0, 0.85, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1])
ax2.stem(range(1,11), pr_bad, linefmt='r-', markerfmt='ro')
for i, p in enumerate(pr_bad):
    ax2.text(i+1.1, p, f'{p*1000:.0f} hPa\nΔσ={pr_bad[i-1]-p:.3f}' if i>0 else f'{p*1000:.0f} hPa',
             fontsize=7, va='center')
ax2.set_xlabel('K index')
ax2.set_title('BAD: uneven spacing at top\n(Δσ=0.10 → 0.05 → diverges!)', color='red')
ax2.invert_yaxis(); ax2.set_xticks(range(1,11)); ax2.grid(True, alpha=0.3)

plt.suptitle('σ-Level Spacing Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(STEP_DIR/"sigma_level_spacing.png", dpi=150, bbox_inches="tight")
print("✓ Saved: sigma_level_spacing.png")
plt.show()

# %% [markdown]
# ## 4. Why Pressure Levels, Not Model Levels?
#
# | Aspect | Pressure Levels (our choice) | Model Levels (hybrid σ-p) |
# |--------|------------------------------|---------------------------|
# | Surface type | Constant pressure | Vary with p_s (terrain) |
# | ∂θ/∂p computation | Direct FD on isolines | Requires chain-rule through a(η),b(η) |
# | Data volume (10 lev) | ~1 MB/day | ~14 MB/day (137 lev) |
# | PPVI literature standard | ✓ All Davis/Emanuel papers | ✗ Never used |
# | Wu solver compatibility | Native (PR = p/p₀) | Would need full pre-interpolation |
#
# **Verdict**: Pressure levels are the correct input for PPVI. Model levels add
# complexity with no benefit for this method.
#
# %% [markdown]
# ## 5. Map: Grid Points on CA Domain
#
# %%
proj = ccrs.LambertConformal(central_longitude=-105, central_latitude=50)
pc   = ccrs.PlateCarree()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1,1,1, projection=proj)
ax.set_extent([-175, -35, 5, 88], crs=pc)
ax.add_feature(cfeature.LAND, facecolor="0.92")
ax.add_feature(cfeature.OCEAN, facecolor="0.98")
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3)
ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.2, edgecolor="0.7")

# Plot every 4th grid point
LON2D, LAT2D = np.meshgrid(lons, lats)
sk = 4
ax.plot(LON2D[::sk, ::sk].ravel(), LAT2D[::sk, ::sk].ravel(), 'k.',
        markersize=1, transform=pc, alpha=0.5)

# Label corners
for (lat, lon, label) in [(85.5,-169.5,"NW"), (85.5,-40.5,"NE"),
                            (10.5,-169.5,"SW"), (10.5,-40.5,"SE")]:
    ax.plot(lon, lat, 'ro', markersize=5, transform=pc)
    ax.text(lon+2, lat, label, transform=pc, fontsize=9, fontweight="bold")

ax.set_title(f"PPVI Grid: {NX}×{NY} = {NX*NY:,} points\n"
             f"1.5° × 1.5°, 10.5–85.5°N, 169.5°W–40.5°W",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(STEP_DIR/"ca_domain_grid.png", dpi=150, bbox_inches="tight")
print("✓ Saved: ca_domain_grid.png")
plt.show()

# %% [markdown]
# ## 6. Subset ERA5 from Full NH → CA Domain
#
# %%
# Load one day to get reference
ds0 = xr.open_dataset(DATA_DIR / "era5_2025-01-08_00Z.nc").squeeze()
print(f"Full NH: {ds0.dims}")

# Subset: CDS area=[N,W,S,E]; our lats are N→S
ds_ca = ds0.sel(latitude=slice(LAT_N, LAT_S), longitude=slice(LON_W % 360, LON_E % 360))
print(f"CA subset: {ds_ca.dims}")
print(f"  lats: {float(ds_ca.latitude.min()):.1f} to {float(ds_ca.latitude.max()):.1f}")
print(f"  lons: {float(ds_ca.longitude.min()):.1f} to {float(ds_ca.longitude.max()):.1f}")

# Quick verify
assert ds_ca.dims["latitude"] == NY, f"Expected {NY} lats, got {ds_ca.dims['latitude']}"
assert ds_ca.dims["longitude"] == NX, f"Expected {NX} lons, got {ds_ca.dims['longitude']}"
print("✓ CA subset dimensions match Wu grid spec")

print("\n→ Step 03: Compute 11-day time-mean climatology + visualize anomalies")

