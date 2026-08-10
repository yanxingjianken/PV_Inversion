# %% [markdown]
# Step 11 — Plot C: Linear Closure Check
#
# Verifies that the piecewise PV inversion satisfies linear closure:
# **Σ(pieces) = total perturbation** — i.e., the sum of the 3 piecewise
# solutions equals the perturbation solved from the full PV anomaly.
#
# **Two references:**
# - **Ref A**: Single all-levels BALP inversion (run qinvert​p21 with NOUT=1,
#   piece spanning levels 1–9). Exact test of solver linearity.
# - **Ref B**: observed ψ′ from inverting ζ′ of (u_event−u_clim, v_event−v_clim).
#   Tests reconstruction of the real balanced perturbation.
#
# **Output figure (6 panels + scatter + profile):**
# - Row 1 (Ref A): Sum-of-pieces ψ′, total ψ′, residual (250 hPa)
# - Row 2 (Ref B): Same as Row 1
# - Row 3: Scatter Σ vs Total (250 hPa) + vertical profile of domain-RMS residual
#
# Reads: `piecewise_psi.nc`, `event_pert.out`, `event_bal.out`, `meanh`
#
# %% [markdown]
# ## 1. Load Data & Parse
#
# %%
import numpy as np, xarray as xr, matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
import matplotlib.ticker as mticker
from pathlib import Path
import subprocess, os, yaml

import sys
_sys_path_root = str(Path(__file__).resolve().parent.parent.parent)
if _sys_path_root not in sys.path:
    sys.path.insert(0, _sys_path_root)
import config

STEP_DIR = Path(__file__).resolve().parent
OUT_DIR = Path(config.WU_OUT_DIR)
WU_DIR = Path(config.WU_IN_DIR)
CLIM_DIR = Path(config.CLIM_DIR)
BUILD_DIR = Path(config.DATA_DIR) / "wu_bin"
_YAML_CFG_PATH = Path(_sys_path_root) / "wu_config.yaml"
with open(_YAML_CFG_PATH) as _f:
    _yaml_cfg = yaml.safe_load(_f)
_pd = _yaml_cfg["pass_d"]

NW, NY, NX = config.NW, config.NY, config.NX
NPIECES, block = config.NPIECES, NY * NX
PLEVS = config.PLEVS
K250, K500 = 6, 3

# --- Load piecewise data: parse SP directly from event_pert.out ---
# (HP in piecewise_psi.nc may be stale — read raw Fortran output instead)
ds_psi = xr.open_dataset(OUT_DIR / "piecewise_psi.nc")  # for lat/lon only
lats = ds_psi["lat"].values
lons = ds_psi["lon"].values
LON2D, LAT2D = np.meshgrid(lons, lats)
ds_psi.close()

def read_all_wu(fp):
    """Read all floats from Fortran-ascii output."""
    data = []
    with open(fp) as f:
        for line in f:
            for tok in line.split():
                data.append(float(tok))
    return np.array(data)

_vp_raw = read_all_wu(WU_DIR / "event_pert.out")
SP = np.zeros((NPIECES, NW, NY, NX))
pp_raw = 2 * NW * block  # 1 piece: NW HP + NW SP
for ip in range(NPIECES):
    base = ip * pp_raw + NW * block  # skip HP, read SP
    for k in range(NW):
        SP[ip, k] = _vp_raw[base + k*block: base + (k+1)*block].reshape(NY, NX)
# Actual ψ′ in m²/s: SP is non-dim ψ′ ÷ 1e5
PSI_pieces = SP * 1.0e5  # (3, 9, 51, 87)

# Sum of piecewise ψ′
PSI_sum = np.sum(PSI_pieces, axis=0)  # (9, 51, 87)


# %% [markdown]
# ## 2. Reference A — Single All-Levels BALP Inversion
#
# %%
def read_wu_ascii(fp):
    """Read Fortran-ascii output (first line header skipped by format)."""
    data = []
    with open(fp) as f:
        for line in f:
            for tok in line.split():
                data.append(float(tok))
    return np.array(data)


# Check if event_total.out already exists (skip re-run)
total_out = WU_DIR / "event_total.out"
if not total_out.exists():
    print("Running single all-levels BALP inversion (NOUT=1, piece 1–9)...")
    os.chdir(str(WU_DIR))
    exe = BUILD_DIR / "qinvertp21.exe"
    for fn in ["event_total.out"]:
        (WU_DIR / fn).unlink(missing_ok=True)

    _hout_list = ",".join(str(i) for i in range(1, NW + 1))
    _siout_list = _hout_list

    stdin_total = f"""{_pd['omegs']}
{_pd['omegh']}
{_pd['part']}
{_pd['thrsh']}
{_pd['tscal']}
{_pd['qscal']}
'meanq'
'meanh'
'event_q.out'
'event_bal.out'
'event_total.out'
{_pd['imap']}
0
{_pd['iqd']}
{NW},{_hout_list}
{NW},{_siout_list}
1
{NW},{','.join(str(k) for k in range(1, NW+1))}
0
"""

    result_t = subprocess.run(
        [str(exe)], input=stdin_total, capture_output=True, text=True, timeout=600
    )
    for line in result_t.stdout.split("\n")[-15:]:
        if line.strip():
            print(f"  {line}")
    print(f"  Return code: {result_t.returncode}")
else:
    print(f"event_total.out already exists ({total_out.stat().st_size} bytes) — skipping Fortran run.")

# Parse total perturbation ψ′ from event_total.out
_vp_total = read_wu_ascii(WU_DIR / "event_total.out")
pp_total = 2 * NW * block  # 1 piece: NW HP + NW SP
PSI_total_A = np.zeros((NW, NY, NX))
for k in range(NW):
    base_total = NW * block + k * block  # skip HP block, read SP block
    PSI_total_A[k] = _vp_total[base_total: base_total + block].reshape(NY, NX) * 1.0e5

# %% [markdown]
# ## 3. Reference B — Observed ψ′ from event−clim winds
#
# ψ′ is recovered from the *observed* wind anomaly by inverting the relative
# vorticity of (u_event−u_clim, v_event−v_clim):
#     ∇²ψ′ = ζ′ = ∂v′/∂x − ∂u′/∂y .
# This is the "truth" the piecewise PV inversion should reproduce.
#
# %%
import xarray as xr
import pvtend.helmholtz as hel

# 1. Load Global Event Data & extract global psi
_ds_event_global = xr.open_dataset("/net/flood/data2/users/x_yan/pv_ertel_compute/test00_era5_sanity_check/data/era5_2025-01-08_00z_pl.nc")
dim_p = "pressure_level" if "pressure_level" in _ds_event_global.dims else "level"
_ds_event_global = _ds_event_global.sel({dim_p: PLEVS})
_ds_event_global = _ds_event_global.isel(valid_time=0) if "valid_time" in _ds_event_global.dims else _ds_event_global.isel(time=0)
event_lat = _ds_event_global.latitude.values
event_lon = _ds_event_global.longitude.values

if event_lat[0] > event_lat[-1]:
    event_lat = event_lat[::-1]
    u_event_global = _ds_event_global["u"].values[:, ::-1, :]
    v_event_global = _ds_event_global["v"].values[:, ::-1, :]
else:
    u_event_global = _ds_event_global["u"].values
    v_event_global = _ds_event_global["v"].values

psi_event_global = hel.helmholtz_decomposition_3d(u_event_global, v_event_global, event_lat, event_lon)['psi']
da_psi_event = xr.DataArray(psi_event_global, coords=[PLEVS, event_lat, event_lon], dims=["level", "lat", "lon"])
# Keep native 0-360 longitudes (global ERA5 is 0..358.5 monotonic) so the
# subsequent interp onto the 0-360 regional domain (120..319.5) stays in range.
da_psi_event = da_psi_event.sortby("lon")

# 2. Load NH Climatology Rotational Data & extract NH psi
_ds_base_clim = xr.open_dataset("/net/flood/data2/users/x_yan/era/clim/era5_hourly_clim_1990-2020_jan_u.nc")
clim_lat = _ds_base_clim.latitude.values
clim_lon = _ds_base_clim.longitude.values

_ds_clim_u = xr.open_dataset("/net/flood/data2/users/x_yan/era/clim/era5_hourly_clim_1990-2020_jan_u_helmholtz.nc")
_ds_clim_v = xr.open_dataset("/net/flood/data2/users/x_yan/era/clim/era5_hourly_clim_1990-2020_jan_v_helmholtz.nc")

u_rot_clim = _ds_clim_u["u_rot_bar"].isel(day=7, hour=0, pressure_level=slice(0, NW)).values
v_rot_clim = _ds_clim_v["v_rot_bar"].isel(day=7, hour=0, pressure_level=slice(0, NW)).values

if clim_lat[0] > clim_lat[-1]:
    clim_lat = clim_lat[::-1]
    u_rot_clim = u_rot_clim[:, ::-1, :]
    v_rot_clim = v_rot_clim[:, ::-1, :]

psi_clim_nh = hel.helmholtz_decomposition_3d(u_rot_clim, v_rot_clim, clim_lat, clim_lon)['psi']
da_psi_clim = xr.DataArray(psi_clim_nh, coords=[PLEVS, clim_lat, clim_lon], dims=["level", "lat", "lon"])
# The clim winds are stored on a -180/180 longitude axis (-180..178.5). The
# regional target grid is 0-360 (120..319.5), so interpolating onto lon>180
# would fall OUT OF RANGE and return NaN over the eastern 2/3 of the domain
# (the blank middle/right panels of Row 2). Convert clim ψ to 0-360 monotonic
# first so the interp covers the whole CONUS/Atlantic sector.
da_psi_clim = da_psi_clim.assign_coords(lon=(da_psi_clim.lon % 360)).sortby("lon")

# 3. Interpolate PSI to Regional Domain
psi_ev_reg = da_psi_event.interp(lat=lats, lon=lons, method="cubic").values
psi_cl_reg = da_psi_clim.interp(lat=lats, lon=lons, method="cubic").values

PSI_total_B = psi_ev_reg - psi_cl_reg

# %% [markdown]
# ## 4. Compute Residuals & Statistics
#
# %%
# Residuals at each level
residual_A = PSI_sum - PSI_total_A  # (9, 51, 87)
residual_B = PSI_sum - PSI_total_B

# RMS per level (domain-wide)
rms_A_per_level = np.sqrt(np.nanmean(residual_A**2, axis=(1, 2)))
rms_B_per_level = np.sqrt(np.nanmean(residual_B**2, axis=(1, 2)))

# Normalized RMS: residual / RMS(signal)
rms_signal_sum = np.sqrt(np.nanmean(PSI_sum**2, axis=(1, 2)))
rms_signal_A = np.sqrt(np.nanmean(PSI_total_A**2, axis=(1, 2)))
rms_signal_B = np.sqrt(np.nanmean(PSI_total_B**2, axis=(1, 2)))
nrms_A = rms_A_per_level / np.maximum(rms_signal_sum, 1e-12)
nrms_B = rms_B_per_level / np.maximum(rms_signal_B, 1e-12)

print("\n=== Closure Statistics ===")
print(f"{'Level':>6s}  {'RMS(A)':>10s}  {'NRMS(A)':>8s}  {'RMS(B)':>10s}  {'NRMS(B)':>8s}")
for k in range(NW):
    print(f"{PLEVS[k]:6.0f}  {rms_A_per_level[k]:10.2e}  {nrms_A[k]:8.4f}  "
          f"{rms_B_per_level[k]:10.2e}  {nrms_B[k]:8.4f}")

# R² at 250 and 500 hPa
def r2_score(y_true, y_pred):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    ss_res = np.nansum((y_true[valid] - y_pred[valid]) ** 2)
    ss_tot = np.nansum((y_true[valid] - np.nanmean(y_true[valid])) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-30)

for label, total, residual in [
    ("Ref A (single BALP)", PSI_total_A, residual_A),
    ("Ref B (obs winds)", PSI_total_B, residual_B),
]:
    for lev_name, k in [("250 hPa", K250), ("500 hPa", K500)]:
        r2_k = r2_score(total[k].ravel(), PSI_sum[k].ravel())
        rms_k = np.sqrt(np.nanmean(residual[k]**2))
        print(f"  {label} @ {lev_name}: R²={r2_k:.6f}  RMS residual={rms_k:.2e} m²/s")


# %% [markdown]
# ## 5. Plot — 6-Panel Closure Map + Scatter + Vertical Profile
#
# %%
proj = ccrs.PlateCarree(central_longitude=200)  # center on Pacific domain (no 180° seam)
pc = ccrs.PlateCarree()
fig = plt.figure(figsize=(20, 18))

# ---- Row 1: Reference A (single BALP) ----
titles_A = ["Σ pieces ψ′ (250 hPa)", "Total ψ′ (single BALP)", "Residual (Σ − Total)"]
data_A = [PSI_sum[K250], PSI_total_A[K250], residual_A[K250]]

for col in range(3):
    ax = fig.add_subplot(3, 3, col + 1, projection=proj)
    ax.set_extent([config.LON_W - 2, config.LON_E + 2,
                   config.LAT_S - 2, config.LAT_N + 2], crs=pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.5, edgecolor="gray")
    ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="lightgray")
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.2, edgecolor="lightgray")
    gl = ax.gridlines(draw_labels=(col == 0), lw=0.3, color="lightgray",
                      alpha=0.6, ls="--")
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(-170, -30, 30))
    gl.ylocator = mticker.FixedLocator(range(20, 90, 20))
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    field = data_A[col]
    vm = float(np.nanpercentile(np.abs(field), 99))
    if not np.isfinite(vm) or vm <= 0:
        vm = 1e6
    cf = ax.pcolormesh(LON2D, LAT2D, field, cmap="RdBu_r",
                       vmin=-vm, vmax=vm, transform=pc)
    cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.06,
                        fraction=0.04, shrink=0.80)
    cbar.ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")

    if col == 0:
        ax.set_ylabel("Ref A\n(single BALP)", fontsize=10, fontweight="bold", labelpad=40)
    ax.set_title(titles_A[col], fontsize=9)
# ---- Row 2: Reference B (observed winds) ----
titles_B = ["Σ pieces ψ′ (250 hPa)", "Obs ψ′ (event−clim winds)", "Residual (Σ − Obs)"]
data_B = [PSI_sum[K250], PSI_total_B[K250], residual_B[K250]]

for col in range(3):
    ax = fig.add_subplot(3, 3, col + 4, projection=proj)
    ax.set_extent([config.LON_W - 2, config.LON_E + 2,
                   config.LAT_S - 2, config.LAT_N + 2], crs=pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.5, edgecolor="gray")
    ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="lightgray")
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.2, edgecolor="lightgray")
    gl = ax.gridlines(draw_labels=(col == 0), lw=0.3, color="lightgray",
                      alpha=0.6, ls="--")
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(-170, -30, 30))
    gl.ylocator = mticker.FixedLocator(range(20, 90, 20))
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    field = data_B[col]
    vm = float(np.nanpercentile(np.abs(field), 99))
    if not np.isfinite(vm) or vm <= 0:
        vm = 1e6
    cf = ax.pcolormesh(LON2D, LAT2D, field, cmap="RdBu_r",
                       vmin=-vm, vmax=vm, transform=pc)
    cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.06,
                        fraction=0.04, shrink=0.80)
    cbar.ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")

    if col == 0:
        ax.set_ylabel("Ref B\n(obs winds)", fontsize=10, fontweight="bold", labelpad=40)
    ax.set_title(titles_B[col], fontsize=9)

# ---- Row 3, Left: Scatter Σ vs Total @ 250 hPa ----
ax_sc = fig.add_subplot(3, 3, 7)
v1d_A = PSI_total_A[K250].ravel()
v1d_B = PSI_total_B[K250].ravel()
v1d_sum = PSI_sum[K250].ravel()
valid_A = np.isfinite(v1d_A) & np.isfinite(v1d_sum)
valid_B = np.isfinite(v1d_B) & np.isfinite(v1d_sum)

vmin_all = min(np.nanmin(v1d_A[valid_A]), np.nanmin(v1d_B[valid_B]),
               np.nanmin(v1d_sum))
vmax_all = max(np.nanmax(v1d_A[valid_A]), np.nanmax(v1d_B[valid_B]),
               np.nanmax(v1d_sum))

ax_sc.scatter(v1d_A[valid_A], v1d_sum[valid_A], s=0.5, alpha=0.3, color="C0",
              label=f"Ref A: R²={r2_score(v1d_A, v1d_sum):.4f}")
ax_sc.scatter(v1d_B[valid_B], v1d_sum[valid_B], s=0.5, alpha=0.3, color="C1",
              label=f"Ref B: R²={r2_score(v1d_B, v1d_sum):.4f}")
ax_sc.plot([vmin_all, vmax_all], [vmin_all, vmax_all], "k--", lw=1, label="1:1")
ax_sc.set_xlabel("Total ψ′ [m²/s]")
ax_sc.set_ylabel("Σ pieces ψ′ [m²/s]")
ax_sc.set_title("Σ pieces vs Total @ 250 hPa", fontsize=9)
ax_sc.legend(fontsize=7, loc="upper left")
ax_sc.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")

# ---- Row 3, Right: Vertical profile of normalized RMS residual ----
ax_vp = fig.add_subplot(3, 3, 8)
ax_vp.plot(nrms_A * 100, PLEVS, "o-", color="C0", label="Ref A (single BALP)", lw=2)
ax_vp.plot(nrms_B * 100, PLEVS, "s--", color="C1", label="Ref B (obs winds)", lw=2)
ax_vp.set_ylim(1000, 100)
ax_vp.set_yticks(PLEVS)
ax_vp.set_ylabel("Pressure [hPa]")
ax_vp.set_xlabel("Normalized RMS residual [%]")
ax_vp.set_title("Vertical Profile: nRMS Residual", fontsize=9)
ax_vp.legend(fontsize=7)
ax_vp.grid(True, alpha=0.3)
# Add reference vertical line at 0
ax_vp.axvline(x=0, color="gray", lw=0.5, ls=":")

# ---- Row 3, Far Right: Absolute RMS residual vertical profile ----
ax_rms = fig.add_subplot(3, 3, 9)
ax_rms.plot(rms_A_per_level, PLEVS, "o-", color="C0", label="Ref A", lw=2)
ax_rms.plot(rms_B_per_level, PLEVS, "s--", color="C1", label="Ref B", lw=2)
ax_rms.set_ylim(1000, 100)
ax_rms.set_yticks(PLEVS)
ax_rms.set_xlabel("RMS residual [m²/s]")
ax_rms.set_title("Vertical Profile: RMS Residual", fontsize=9)
ax_rms.legend(fontsize=7)
ax_rms.grid(True, alpha=0.3)
ax_rms.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")

fig.suptitle(
    "Linear Closure Check: Σ(Piecewise ψ′) vs Total Perturbation ψ′\n"
    "CA Blocking Event 2025-01-08 00Z  |  INLIN=0  |  8 Levels  |  30yr Clim",
    fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])

fname_png = STEP_DIR / "closure_check.png"
fname_pdf = STEP_DIR / "closure_check.pdf"
fig.savefig(fname_png, dpi=200, bbox_inches="tight")
fig.savefig(fname_pdf, bbox_inches="tight")
print(f"\n✓ Saved: {fname_png}")
print(f"✓ Saved: {fname_pdf}")
plt.close(fig)

print("\n=== Plot 2 complete — closure_check ===")
