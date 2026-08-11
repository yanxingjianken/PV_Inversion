#!/usr/bin/env python3
"""Diagnose why Ref B (BALP ψ′ vs observed ψ′) doesn't close.

The two "ψ′" quantities come from fundamentally different operators:
  - BALP ψ′: 3D nonlinear PV inversion (∇·(f∇ψ) + 2J + 3D PV coupling)
  - Obs ψ′:  2D Poisson inversion of local ζ′ = ∂v'/∂x − ∂u'/∂y from total winds

Key tests:
  H1: Is the pattern correct even if amplitude is wrong? → spatial correlation
  H2: Is there a constant scale factor? → regression slope
  H3: Is the BALP mean ψ consistent with obs? → compare BALP mean vs Poisson(ζ_clim)
  H4: What's the 3D PV coupling effect? → does BALP differ from local-ζ inversion?
"""
import numpy as np, xarray as xr, matplotlib.pyplot as plt
from pathlib import Path
from scipy.fft import dstn, idstn
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

WU_DIR = Path(config.WU_IN_DIR)
CLIM_DIR = Path(config.CLIM_DIR)
NW, NY, NX = config.NW, config.NY, config.NX
NPIECES, block = config.NPIECES, NY * NX
PLEVS = config.PLEVS
K250, K500 = 6, 3

# ── Helpers ──
def read_all_wu(fp):
    data = []
    with open(fp) as f:
        for line in f:
            for tok in line.split():
                data.append(float(tok))
    return np.array(data)

def poisson_dst(rhs, dx, dy):
    f_int = rhs[1:-1, 1:-1]
    my, mx = f_int.shape
    F = dstn(f_int, type=1)
    jy = np.arange(1, my+1)[:, None]
    ix = np.arange(1, mx+1)[None, :]
    lam = (2*np.cos(jy*np.pi/(my+1))-2)/dy**2 + (2*np.cos(ix*np.pi/(mx+1))-2)/dx**2
    psi_int = idstn(F / lam, type=1)
    psi = np.zeros_like(rhs)
    psi[1:-1, 1:-1] = psi_int
    return psi

def r2(y_true, y_pred):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    ss_res = np.nansum((y_true[valid]-y_pred[valid])**2)
    ss_tot = np.nansum((y_true[valid]-np.nanmean(y_true[valid]))**2)
    return 1.0 - ss_res / max(ss_tot, 1e-30)

def slope(y_true, y_pred):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    return np.polyfit(y_pred[valid].ravel(), y_true[valid].ravel(), 1)[0]

# ── 1. Load BALP ψ' (Σ pieces) ──
_vp_raw = read_all_wu(WU_DIR / "event_pert.out")
SP = np.zeros((NPIECES, NW, NY, NX))
pp_raw = 2 * NW * block
for ip in range(NPIECES):
    base = ip * pp_raw + NW * block
    for k in range(NW):
        SP[ip, k] = _vp_raw[base + k*block: base + (k+1)*block].reshape(NY, NX)
PSI_balp = np.sum(SP, axis=0) * 1.0e5  # Σ pieces ψ′ in m²/s

# ── 2. Load mean BALP ψ (from meanh) ──
_vmh = read_all_wu(WU_DIR / "meanh")
PSI_balp_mean = np.zeros((NW, NY, NX))
for k in range(NW):
    PSI_balp_mean[k] = _vmh[NW*block + k*block: NW*block + (k+1)*block].reshape(NY, NX) * 1.0e5

# ── 3. Load obs winds, compute ζ and ψ ──
ds_mean = xr.open_dataset(CLIM_DIR / "mean_clim.nc")
ds_event = xr.open_dataset(CLIM_DIR / "event.nc")
lats = ds_mean.latitude.values
lons = ds_mean.longitude.values

U_mean = ds_mean["u"].values; V_mean = ds_mean["v"].values
U_event = ds_event["u"].values; V_event = ds_event["v"].values
ds_mean.close(); ds_event.close()

U_anom = U_event - U_mean
V_anom = V_event - V_mean

R_E = 6.371e6
dlat_rad_arr = np.deg2rad(np.gradient(lats))[None, :, None]
dlon_rad_arr = np.deg2rad(np.gradient(lons))[None, None, :]
coslat_arr = np.cos(np.deg2rad(lats))[None, :, None]

def compute_zeta(u, v):
    dvdx = np.gradient(v, axis=2) / (R_E * coslat_arr * dlon_rad_arr)
    dudy = np.gradient(u, axis=1) / (R_E * dlat_rad_arr)
    return dvdx - dudy

ZETA_mean = compute_zeta(U_mean, V_mean)
ZETA_event = compute_zeta(U_event, V_event)
ZETA_anom = compute_zeta(U_anom, V_anom)

dx_m = R_E * np.cos(np.deg2rad(np.mean(lats))) * np.deg2rad(config.DLON)
dy_m = R_E * np.deg2rad(config.DLAT)
PSI_obs_mean = np.array([poisson_dst(ZETA_mean[k], dx_m, dy_m) for k in range(NW)])
PSI_obs_anom = np.array([poisson_dst(ZETA_anom[k], dx_m, dy_m) for k in range(NW)])

# ── 4. Diagnostics ──
print("=" * 70)
print("H1: Amplitude comparison")
print(f"  BALP Σ-pieces ψ′ @250: [{PSI_balp[K250].min():.2e}, {PSI_balp[K250].max():.2e}]")
print(f"  Poisson(ζ′_total)  @250: [{PSI_obs_anom[K250].min():.2e}, {PSI_obs_anom[K250].max():.2e}]")
print(f"  Ratio rms: {np.sqrt(np.mean(PSI_balp[K250]**2)) / np.sqrt(np.mean(PSI_obs_anom[K250]**2)):.2f}")

print(f"\n  BALP mean ψ @250: [{PSI_balp_mean[K250].min():.2e}, {PSI_balp_mean[K250].max():.2e}]")
print(f"  Poisson(ζ_mean) @250: [{PSI_obs_mean[K250].min():.2e}, {PSI_obs_mean[K250].max():.2e}]")
print(f"  Ratio rms (mean): {np.sqrt(np.mean(PSI_balp_mean[K250]**2)) / np.sqrt(np.mean(PSI_obs_mean[K250]**2)):.2f}")

print("\n" + "=" * 70)
print("H2: Per-level R², slope, spatial correlation (BALP Σ-pieces vs Poisson(ζ′_total))")
print(f"{'Level':>6s}  {'R²':>10s}  {'Slope':>10s}  {'Corr':>10s}  {'RMS(BALP)':>12s}  {'RMS(obs)':>12s}")
for k in range(NW):
    r2_v = r2(PSI_balp[k], PSI_obs_anom[k])
    sl = slope(PSI_balp[k], PSI_obs_anom[k])
    corr = np.corrcoef(PSI_balp[k].ravel(), PSI_obs_anom[k].ravel())[0,1]
    rms_b = np.sqrt(np.mean(PSI_balp[k]**2))
    rms_o = np.sqrt(np.mean(PSI_obs_anom[k]**2))
    print(f"{PLEVS[k]:6.0f}  {r2_v:10.4f}  {sl:10.3f}  {corr:10.4f}  {rms_b:12.2e}  {rms_o:12.2e}")

print("\n" + "=" * 70)
print("H3: If we scale Poisson(ζ′) to match BALP amplitude (multiply by slope),")
print("     does R² improve? (Tests if problem is just amplitude, not pattern)")
for k in range(NW):
    sl = slope(PSI_balp[k], PSI_obs_anom[k])
    PSI_scaled = PSI_obs_anom[k] * sl
    r2_scaled = r2(PSI_balp[k], PSI_scaled)
    corr = np.corrcoef(PSI_balp[k].ravel(), PSI_obs_anom[k].ravel())[0,1]
    print(f"  {PLEVS[k]:4.0f} hPa: slope={sl:.3f}, corr={corr:.4f}, R²(scaled)={r2_scaled:.4f}")

print("\n" + "=" * 70)
print("H4: BALP mean ψ vs Poisson(ζ_clim) — test if BALP ≈ Poisson for MEAN state")
print("     (mean flow is closer to linear, QG balance)")
for k in range(NW):
    corr_m = np.corrcoef(PSI_balp_mean[k].ravel(), PSI_obs_mean[k].ravel())[0,1]
    sl_m = slope(PSI_balp_mean[k], PSI_obs_mean[k])
    r2_m = r2(PSI_balp_mean[k], PSI_obs_mean[k])
    print(f"  {PLEVS[k]:4.0f} hPa: corr={corr_m:.4f}, slope={sl_m:.3f}, R²={r2_m:.4f}")

# ── 5. Quick plot ──
LON2D, LAT2D = np.meshgrid(lons, lats)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for col, (data, title) in enumerate([
    (PSI_balp_mean[K500], "BALP mean ψ @500"),
    (PSI_obs_mean[K500], "Poisson(ζ_clim) ψ @500"),
    (PSI_balp_mean[K500] - PSI_obs_mean[K500], "BALP − Poisson mean @500"),
]):
    vm = float(np.nanpercentile(np.abs(data), 99))
    cf = axes[0, col].pcolormesh(LON2D, LAT2D, data, cmap="RdBu_r", vmin=-vm, vmax=vm)
    plt.colorbar(cf, ax=axes[0, col], orientation="horizontal", pad=0.08, fraction=0.04)
    axes[0, col].set_title(title, fontsize=10)

sl_250 = slope(PSI_balp[K250], PSI_obs_anom[K250])
for col, (data, title) in enumerate([
    (PSI_balp[K250], "BALP Σ-pieces ψ′ @250"),
    (PSI_obs_anom[K250], "Poisson(ζ′) ψ′ @250"),
    (PSI_balp[K250] - PSI_obs_anom[K250] * sl_250,
     f"BALP − {sl_250:.1f}×Poisson @250"),
]):
    vm = float(np.nanpercentile(np.abs(data), 99))
    cf = axes[1, col].pcolormesh(LON2D, LAT2D, data, cmap="RdBu_r", vmin=-vm, vmax=vm)
    plt.colorbar(cf, ax=axes[1, col], orientation="horizontal", pad=0.08, fraction=0.04)
    axes[1, col].set_title(title, fontsize=10)

fig.suptitle("BALP ψ vs Poisson(ζ) ψ — Mean & Perturbation", fontsize=13, fontweight="bold")
out_path = Path(__file__).resolve().parent / "closure_diagnostic.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n✓ Saved: {out_path}")
