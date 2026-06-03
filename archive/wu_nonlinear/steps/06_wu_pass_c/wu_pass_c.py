# %% [markdown]
# Step 06 — Wu Pass C: Total Balanced PV Inversion (qinvert21)
#
# The `qinvert21` program solves the full coupled nonlinear balance:
# - 2-D Helmholtz for ψ on each σ-level
# - 3-D PV–streamfunction relation with gradient-wind balance
# - SOR Picard iteration with under-relaxation between ψ and Φ
#
# **Params driven by**: `wu/wu_config.yaml`
#
# %% [markdown]
# ## 1. Run qinvert21
#
# %%
import subprocess, os, numpy as np, matplotlib.pyplot as plt, yaml
import cartopy.crs as ccrs, cartopy.feature as cfeature
from pathlib import Path

import sys; from pathlib import Path as _Path
_sys_path_root = str(_Path(__file__).resolve().parent.parent.parent)
if _sys_path_root not in sys.path: sys.path.insert(0, _sys_path_root)
import config
_YAML_CFG_PATH = _Path(_sys_path_root) / "wu_config.yaml"
with open(_YAML_CFG_PATH) as _f:
    _yaml_cfg = yaml.safe_load(_f)
_pc = _yaml_cfg["pass_c"]
STEP_DIR = _Path(__file__).resolve().parent
DATA_DIR = _Path(config.DATA_DIR); ERA5_DIR = _Path(config.ERA5_DIR)
CLIM_DIR = _Path(config.CLIM_DIR); WU_DIR = _Path(config.WU_IN_DIR)
OUT_DIR = _Path(config.WU_OUT_DIR); FIG_DIR = _Path(config.FIG_DIR)
BUILD_DIR = _Path(config.DATA_DIR) / "wu_bin"

# (STEP_DIR from config import)
# (WU_DIR from config)
# (BUILD_DIR from config import)

# Pass C stdin — driven by wu_config.yaml
os.chdir(str(WU_DIR))
exe = BUILD_DIR / "qinvert21.exe"
for fn in ["event_bal.out"]:
    (WU_DIR / fn).unlink(missing_ok=True)

stdin_c = f"""{_pc['max_iter']}
{_pc['max_outer']}
{_pc['omegs']}
{_pc['omegh']}
{_pc['part']}
{_pc['thrsh']}
'event_h.out'
'event_q.out'
'event_bal.out'
{_pc['imap']}
{_pc['qmin']}
{_pc['inf']}
"""

result_c = subprocess.run(
    [str(exe)], input=stdin_c, capture_output=True, text=True, timeout=300
)
print("Pass C stdout (last 20 lines):")
for line in result_c.stdout.split("\n")[-20:]:
    print(f"  {line}")
if result_c.returncode != 0:
    print(f"ERROR stderr: {result_c.stderr}")

fp = WU_DIR / "event_bal.out"
if fp.exists():
    print(f"\n✓ event_bal.out created ({fp.stat().st_size} bytes)")
else:
    print("\n✗ event_bal.out MISSING!")
os.chdir(str(STEP_DIR))

# %% [markdown]
# ## 2. Read & Parse event_bal.out
#
# %%
def read_wu_ascii(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            for tok in line.split():
                data.append(float(tok))
    return np.array(data[:8]), np.array(data[8:])

NX, NY, NW = 52, 31, 10
block = NY * NX

hdr_bal, vals_bal = read_wu_ascii(WU_DIR / "event_bal.out")
total = len(vals_bal)
print(f"event_bal.out: {total} values, expected 2*{NW}*{block}={2*NW*block}")
n_blocks = total // block
print(f"Blocks: {n_blocks} (expecting {2*NW})")

# event_bal.out: NW levels Φ, then NW levels ψ (Wu convention)
H_bal = np.zeros((NW, NY, NX))
PSI_bal = np.zeros((NW, NY, NX))
for k in range(NW):
    H_bal[k] = vals_bal[k*block:(k+1)*block].reshape(NY, NX)
for k in range(NW):
    PSI_bal[k] = vals_bal[(NW+k)*block:(NW+k+1)*block].reshape(NY, NX)

print(f"H_bal range:  [{H_bal.min():.2f}, {H_bal.max():.2f}]")
print(f"ψ_bal range:  [{PSI_bal.min():.4f}, {PSI_bal.max():.4f}] (÷1e5)")

# Also read event_h.out for comparison (Pass B ψ)
hdr_h, vals_h = read_wu_ascii(WU_DIR / "event_h.out")
PSI_passb = np.zeros((NW, NY, NX))
for k in range(NW):
    PSI_passb[k] = vals_h[(NW+k)*block:(NW+k+1)*block].reshape(NY, NX)

# %% [markdown]
# ## 3. Visualize: ψ at 500 hPa — Pass B vs Pass C (Refinement)
#
# %%
lats = 85.5 - np.arange(NY) * 1.5
lons = -169.5 + np.arange(NX) * 1.5
LON2D, LAT2D = np.meshgrid(lons, lats)
proj = ccrs.LambertConformal(central_longitude=-105, central_latitude=50)
pc   = ccrs.PlateCarree()

psi_b = PSI_passb[5] * 1.0e5   # Pass B (initial guess)
psi_c = PSI_bal[5] * 1.0e5     # Pass C (refined)
psi_diff = psi_c - psi_b        # refinement

fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": proj})
titles = ["Pass B ψ @ 500 hPa (initial)", "Pass C ψ @ 500 hPa (refined)", "ψ_C − ψ_B (refinement)"]
datas  = [psi_b, psi_c, psi_diff]
cmaps  = ["RdBu_r", "RdBu_r", "RdBu_r"]
vmxs   = [np.percentile(np.abs(psi_b), 99), np.percentile(np.abs(psi_c), 99),
          np.percentile(np.abs(psi_diff), 99)]

for ax, title, data, cmap, vmx in zip(axes, titles, datas, cmaps, vmxs):
    # Guard against a degenerate (all-zero) panel, e.g. when Pass C ψ equals
    # Pass B ψ exactly (case is already balanced) → refinement panel is 0.
    if not np.isfinite(vmx) or vmx <= 0:
        vmx = 1.0
    ax.set_extent([-175, -35, 5, 88], crs=pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.4")
    cf = ax.pcolormesh(LON2D, LAT2D, data, cmap=cmap, transform=pc,
                       vmin=-vmx, vmax=vmx)
    if np.nanmax(np.abs(data)) > 0:
        ax.contour(LON2D, LAT2D, data, colors="black", linewidths=0.3,
                   transform=pc, levels=np.linspace(-vmx, vmx, 14))
    plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02, label="ψ [m²/s]")
    ax.set_title(title, fontsize=10)

plt.suptitle("Wu Pass C — Total Balanced PV Inversion: ψ Refinement\n"
             f"max |ψ_C − ψ_B| = {np.max(np.abs(psi_diff)):.0f} m²/s",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(STEP_DIR/"pass_c_psi_refinement.png", dpi=150, bbox_inches="tight")
print("✓ Saved: pass_c_psi_refinement.png")
plt.show()

# %% [markdown]
# ## 4. Visualize: ψ at All 10 Pressure Levels
#
# %%
PLEV = np.array([1000., 925., 850., 700., 600., 500., 400., 300., 250., 200.])

fig, axes = plt.subplots(2, 5, figsize=(22, 9), subplot_kw={"projection": proj})
for k, ax in enumerate(axes.flat):
    psi_k = PSI_bal[k] * 1.0e5
    vm = np.percentile(np.abs(psi_k), 99)
    ax.set_extent([-175, -35, 5, 88], crs=pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.2, edgecolor="0.5")
    cf = ax.pcolormesh(LON2D, LAT2D, psi_k, cmap="RdBu_r", transform=pc,
                       vmin=-vm, vmax=vm)
    ax.set_title(f"K={k+1}: {PLEV[k]:.0f} hPa\nψ [{psi_k.min():.0f}, {psi_k.max():.0f}] m²/s",
                 fontsize=7)

plt.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.5, pad=0.02, label="ψ [m²/s]")
plt.suptitle("Wu Pass C — Total Balanced ψ at All 10 Pressure Levels [m²/s]",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(STEP_DIR/"pass_c_psi_all_levels.png", dpi=150, bbox_inches="tight")
print("✓ Saved: pass_c_psi_all_levels.png")
plt.show()

# %% [markdown]
# ## 5. Convergence Check
#
# Look for "TOTAL CONVERGENCE" in the Pass C output. If the solver
# reached the maximum iterations without converging, you'll see
# "TOO MANY ITERATIONS" — in that case, increase OMEGS/OMEGH or MAXT.
#
# %%
output_lines = result_c.stdout.split("\n")
converged = any("CONVERGENCE" in line for line in output_lines)
too_many = any("TOO MANY" in line for line in output_lines)
print(f"Converged: {converged} | Exceeded max: {too_many}")
if not converged:
    print("⚠ Pass C did not converge! Check solver parameters.")
elif too_many:
    print("⚠ Pass C hit iteration limit — increase MAXT.")
else:
    print("✓ Pass C converged successfully")

print("\n→ Step 07: Wu Pass D — Piecewise perturbation PV inversion (3 pieces)")

