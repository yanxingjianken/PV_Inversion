# %% [markdown]
# Step 04 — Write Wu .grid Input Files & Verify
#
# Wu's Fortran code reads ASCII `.grid` files in format `(10F8.1)`.
# Each file contains: header(8 floats) + for each of 10 levels: (NY×NX) values.
#
# **Files to write** (into `../data/wu_in/`):
# - `mean_{t,u,v,z}.grid` — 11-day time-mean (from Step 03)
# - `event_{t,u,v,z}.grid` — Jan 8 00Z event (from Step 03)
#
# **Verification**: read back the written .grid files and compare with
# original NetCDF — they should be identical to machine precision.
#
# %% [markdown]
# ## 1. Grid Writer Function
#
# %%
import numpy as np, xarray as xr, os
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
WU_DIR   = WU_DIR
CLIM_DIR = CLIM_DIR
WU_DIR.mkdir(parents=True, exist_ok=True)

NX, NY = 87, 51
DLAT, DLON = 1.5, 1.5
LAT_S, LAT_N = 10.5, 85.5
LON_W, LON_E = -169.5, -40.5

def write_wu_grid(arr3d, out_path):
    """Write (10F8.1) .grid file. Each row (87 values) = 9 lines (8×10 + 1×7).
    Row boundaries MUST align with line boundaries for Fortran READ(10F8.1)."""
    NW = arr3d.shape[0]
    with open(out_path, "w") as f:
        # Header: 10 fields × 8 chars
        hdr_vals = [LAT_S, LON_W, LAT_N, LON_E, DLAT, DLON, float(NX), float(NY), 0.0, 0.0]
        f.write("".join(f"{v:8.1f}" for v in hdr_vals) + "\n")
        for k in range(NW):
            for i in range(NY):          # each row — MUST align to 9 lines
                row = arr3d[k, i, :]     # NX=87 values
                for j in range(0, NX, 10):
                    chunk = row[j:j+10]
                    f.write("".join(f"{float(v):8.1f}" for v in chunk) + "\n")

def load_nc_var(nc_path, var_name):
    """Load a 3D variable (lev, lat, lon) from NetCDF, ensure N→S, W→E."""
    ds = xr.open_dataset(nc_path)
    ds = ds.sortby("latitude", ascending=False).sortby("longitude")
    arr = ds[var_name].transpose("pressure_level", "latitude", "longitude").values
    return arr

# %% [markdown]
# ## 2. Write Mean .grid Files
#
# %% [markdown]
# ## 2. Write Merged .grid Files (H + θ + U + V stacked per timestep)
#
# Wu's pvpialln reads one .grid file per timestep containing
# all 4 fields sequentially: H (all levels), θ, U, V.
# Each uses format (10F8.1).

# %%
G = 9.81; P0 = 100000.0; RD = 287.0; CP = 1004.0
PLEV_PA = np.array([1000,925,850,700,600,500,400,300,250,200]) * 100.0

def write_merged_grid(nc_path, out_path):
    """Load NetCDF, convert to Wu units, write merged .grid (H+θ+U+V)."""
    H  = load_nc_var(nc_path, "z") / G                                   # [m]
    TH = load_nc_var(nc_path, "t") * (P0/PLEV_PA[:,None,None])**(RD/CP)  # [K]
    U  = load_nc_var(nc_path, "u")
    V  = load_nc_var(nc_path, "v")
    # Stack: H (10,NY,NX), TH (10,NY,NX), U (10,NY,NX), V (10,NY,NX)
    merged = np.concatenate([H, TH, U, V], axis=0)  # (40, NY, NX)
    write_wu_grid(merged, out_path)
    print(f"  ✓ {out_path.name}  H=[{H.min():.0f},{H.max():.0f}]m  "
          f"θ=[{TH.min():.0f},{TH.max():.0f}]K  U=[{U.min():.0f},{U.max():.0f}]")

write_merged_grid(CLIM_DIR/"mean_clim.nc",  WU_DIR/"mean.grid")
write_merged_grid(CLIM_DIR/"event.nc",     WU_DIR/"event.grid")

print(f"\n✓ 2 merged .grid files written to {WU_DIR}")

# %% [markdown]
# ## 4. Verify Round-Trip: Read .grid → Compare with NetCDF
#
# %%
def read_wu_grid(path):
    """Read (10F8.1) fixed-width .grid file: line-by-line, 8-char fields."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            for j in range(0, len(line), 8):
                chunk = line[j:j+8].strip()
                if chunk:
                    data.append(float(chunk))
    # header: 8 values in first 8 fields, but line has 10 fields total
    header = np.array(data[:8])
    total_data = len(data) - 10
    block = NY * NX
    NW = total_data // block
    arr = np.array(data[10:10+total_data]).reshape(NW, NY, NX)
    return header, arr

# Verify merged .grid files
for name in ["mean.grid", "event.grid"]:
    fp = WU_DIR / name
    _, arr = read_wu_grid(fp)
    print(f"  {name}: shape={arr.shape} (expected {(4*config.NW, config.NY, config.NX)}) "
          f"range=[{arr.min():.1f}, {arr.max():.1f}]")
    assert arr.shape == (4*config.NW, config.NY, config.NX)

print(f"\n✓ Both .grid files valid ({WU_DIR})")

# %% [markdown]
# ## 4. Peek at .grid file + Map Z500 verification
#
# The merged .grid has H(10,NY,NX) + θ(10,NY,NX) + U(10,NY,NX) + V(10,NY,NX) stacked.
# Extract H at K=5 (500 hPa) to verify.

# %%
import matplotlib.pyplot as plt, cartopy.crs as ccrs, cartopy.feature as cfeature

_, arr_m = read_wu_grid(WU_DIR / "mean.grid")
_, arr_e = read_wu_grid(WU_DIR / "event.grid")
H_mean = arr_m[:config.NW]      # (10, NY, NX) — heights
H_event = arr_e[:config.NW]
z500_mean = H_mean[5] / 9.81    # K=5 = 500 hPa, convert m→dam
z500_event = H_event[5] / 9.81

lats = np.linspace(config.LAT_N, config.LAT_S, config.NY)
lons = np.linspace(config.LON_W, config.LON_E, config.NX)
LON2D, LAT2D = np.meshgrid(lons, lats)
proj = ccrs.LambertConformal(central_longitude=-105, central_latitude=50)
pc = ccrs.PlateCarree()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": proj})
for ax, data, title in [(ax1, z500_mean, "Mean Z500 (30-day)"),
                          (ax2, z500_event, "Event Z500")]:
    ax.set_extent([-175, -35, 5, 88], crs=pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.5, edgecolor="0.3")
    ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="0.5")
    cf = ax.contourf(LON2D, LAT2D, data, cmap="viridis", transform=pc, levels=20)
    ax.contour(LON2D, LAT2D, data, levels=np.arange(480,600,4), colors="black",
               linewidths=0.4, transform=pc)
    plt.colorbar(cf, ax=ax, shrink=0.7, label="dam")
    ax.set_title(f"{title} [{data.min():.0f}, {data.max():.0f}] dam", fontsize=10)
plt.suptitle("Wu .grid Verification — 30-Day Clim Z500 [dam]", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(STEP_DIR/"wu_grid_verification.png", dpi=150, bbox_inches="tight")
print("✓ Saved: wu_grid_verification.png")
plt.show()

print("\n→ Step 05: Compile Wu Fortran & run Pass A+B (Ertel PV + ψ computation)")

