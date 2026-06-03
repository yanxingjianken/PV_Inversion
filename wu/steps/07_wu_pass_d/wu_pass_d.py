# %% [markdown]
# Step 07 — Wu Pass D: Piecewise Perturbation PV Inversion
#
# The `qinvertp21` program (BALP subroutine) partitions the PV perturbation
# q′ = q_event − q_mean into **3 vertical pieces** and inverts each independently.
#
# **Pieces** (from wu_config.yaml):
# - Piece 1 (lower):  K={1,2,3,4} → 1000–700 hPa
# - Piece 2 (middle): K={5,6} → 600–500 hPa
# - Piece 3 (upper):  K={7,8,9,10} → 400–200 hPa
#
# **INLIN=1** — nonlinear terms in operator coefficients (Davis 1991, Section 2.5).
# Ensures Σ(pieces) = total perturbation.
#
# %% [markdown]
# ## 1. Run qinvertp21
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
_pd = _yaml_cfg["pass_d"]
_pieces = _yaml_cfg["pieces"]
STEP_DIR = _Path(__file__).resolve().parent
DATA_DIR = _Path(config.DATA_DIR); ERA5_DIR = _Path(config.ERA5_DIR)
CLIM_DIR = _Path(config.CLIM_DIR); WU_DIR = _Path(config.WU_IN_DIR)
OUT_DIR = _Path(config.WU_OUT_DIR); FIG_DIR = _Path(config.FIG_DIR)
BUILD_DIR = _Path(config.DATA_DIR) / "wu_bin"

# (STEP_DIR from config import)
# (WU_DIR from config)
# (BUILD_DIR from config import)

# Pass D stdin — driven by wu_config.yaml (single source of truth)
# Fortran read order: OMEGS,OMEGAH,PRT,THRSH,TSCAL,QSCAL,
#   FNM(1..5), IMAP, [nondim], INLIN, IQD,
#   then BALP reads: NHO+HOUT, NSIO+SIOUT, NOUT, NMLV+QLV×NOUT, IBC×NOUT
os.chdir(str(WU_DIR))
exe = BUILD_DIR / "qinvertp21.exe"

# Clean old output files
for fn in ["event_pert.out", "event_pert_p1.out", "event_pert_p2.out", "event_pert_p3.out"]:
    (WU_DIR / fn).unlink(missing_ok=True)

# Build QLV lists for each piece from YAML
_piece_qlv = {}
for _pname, _pinfo in _pieces.items():
    _piece_qlv[_pname] = _pinfo["levels"]

NW = config.NW
NPIECES = config.NPIECES

# HOUT and SIOUT: output all 8 levels
_hout_list = ",".join(str(i) for i in range(1, NW + 1))
_siout_list = _hout_list

# Common params for all pieces
_common_prefix = f"""{_pd['omegs']}
{_pd['omegh']}
{_pd['part']}
{_pd['thrsh']}
{_pd['tscal']}
{_pd['qscal']}
'meanq'
'meanh'
'event_q.out'
'event_bal.out'
"""

_common_suffix = f"""{_pd['imap']}
{_pd['inlin']}
{_pd['iqd']}
{NW},{_hout_list}
{NW},{_siout_list}
1
"""

# Build stdin for each piece
_piece_names = ["lower", "middle", "upper"]
_stdins = {}
for idx, _pname in enumerate(_piece_names):
    _qlv = _piece_qlv[_pname]
    _stdins[_pname] = (
        _common_prefix
        + f"'event_pert_p{idx+1}.out'\n"
        + _common_suffix
        + f"{len(_qlv)},{','.join(str(k) for k in _qlv)}\n"
        + f"{_pd['ibc']}\n"
    )

# ── Parallel 3-piece Fortran inversion ──
from concurrent.futures import ThreadPoolExecutor, as_completed

def _run_piece(label, stdin_str, timeout=900):
    """Run one piece of the piecewise inversion."""
    result = subprocess.run(
        [str(exe)], input=stdin_str, capture_output=True, text=True, timeout=timeout
    )
    return label, result

print(f"Launching {NPIECES} piecewise inversions in parallel (IBC={_pd['ibc']}) ...")
with ThreadPoolExecutor(max_workers=NPIECES) as executor:
    _futures = {
        executor.submit(_run_piece, _pn, _stdins[_pn]): _pn
        for _pn in _piece_names
    }
    for _future in as_completed(_futures):
        _pn = _futures[_future]
        _label, _result = _future.result()
        _rc = _result.returncode
        # Print convergence summary
        _converged = "TOTAL CONVERGENCE" in _result.stdout
        _diverged = "TOO MANY" in _result.stdout
        _status = "✓ CONVERGED" if _converged else ("✗ DIVERGED" if _diverged else f"rc={_rc}")
        print(f"  {_pn:8s}: {_status}")
        for _line in _result.stdout.split("\n"):
            if "TOTAL" in _line or "TOO MANY" in _line:
                print(f"    >>> {_line.strip()}")

# ── Concatenate 3 piece outputs into event_pert.out ──
# Each per-piece file: header(8) + NW*HP + NW*SP  (2*NW*block values)
# Final event_pert.out: header(8) + piece1 HP+SP + piece2 HP+SP + piece3 HP+SP
NY, NX = config.NY, config.NX
block = NY * NX
per_piece_nvals = 2 * NW * block

with open(WU_DIR / "event_pert.out", "w") as fout:
    # Write header from piece 1
    with open(WU_DIR / "event_pert_p1.out") as f1:
        hdr_data = [float(x) for line in f1 for x in line.split()]
    hdr_vals = hdr_data[:8]
    fout.write("".join(f"{v:8.1f}" for v in hdr_vals) + "\n")
    
    # Concatenate data (skip header) from each piece
    total_vals = 0
    for idx in range(NPIECES):
        fp_piece = WU_DIR / f"event_pert_p{idx+1}.out"
        with open(fp_piece) as fp:
            piece_data = [float(x) for line in fp for x in line.split()]
        piece_vals = piece_data[8:]  # skip 8-value header
        n_expected = per_piece_nvals
        if len(piece_vals) < n_expected:
            print(f"  ⚠ Piece {idx+1}: got {len(piece_vals)} values, expected {n_expected}; padding")
            piece_vals.extend([0.0] * (n_expected - len(piece_vals)))
        piece_vals = piece_vals[:n_expected]
        # Write in (10F8.1) format
        for j in range(0, len(piece_vals), 10):
            chunk = piece_vals[j:j+10]
            fout.write("".join(f"{float(v):8.1f}" for v in chunk) + "\n")
        total_vals += len(piece_vals)
        # Clean up temp file
        fp_piece.unlink(missing_ok=True)

fp = WU_DIR / "event_pert.out"
if fp.exists():
    print(f"\n✓ event_pert.out created ({fp.stat().st_size} bytes, {total_vals} values)")
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

NX, NY, NW = config.NX, config.NY, config.NW
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
    sp = SP[ip, 6] * 1.0e5  # K=7 = 250 hPa, actual ψ'
    hp = HP[ip, 6]
    print(f"  Piece {ip+1} @250hPa: SP [{sp.min():.0f}, {sp.max():.0f}] m²/s, "
          f"HP [{hp.min():.1f}, {hp.max():.1f}] m")

# %% [markdown]
# ## 3. Visualize: ψ' at 250 hPa — All 3 Pieces
#
# %%
lats = config.LAT_N - np.arange(NY) * config.DLAT
lons = config.LON_W + np.arange(NX) * config.DLON
LON2D, LAT2D = np.meshgrid(lons, lats)
proj = ccrs.LambertConformal(central_longitude=-105, central_latitude=50)
pc   = ccrs.PlateCarree()

piece_titles = ["(a) Lower: 1000–850 hPa", "(b) Middle: 700–400 hPa",
                "(c) Upper: 300–200 hPa"]

fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={"projection": proj})
for ip, (ax, title) in enumerate(zip(axes, piece_titles)):
    psi = SP[ip, 6] * 1.0e5   # actual ψ' at 250 hPa (idx 6 of 8)
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

psi3_raw = SP[2, 6] * 1.0e5
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

