"""A/B compare — Wu (SOR + FD) vs SH (pvtend.sh_ops) Fig-8 panels.

Loads both ``data/wu_out/pv_advection.nc`` and ``data/sh_out/sh_pv_advection.nc``,
plots them side-by-side per piece (2 rows × 3 cols), and prints
quantitative metrics:

* RMSE, pattern correlation, max |Δ| per piece — full patch.
* Same three metrics on the *extratropics-only* subset (20°N ≤ φ ≤ 70°N)
  to isolate the pole-region rigid-lid artefact in Wu's piece 3.
* Same three metrics on the *pole zone* (φ ≥ 70°N) — should show the
  largest SH↔Wu divergence if the rigid-lid hypothesis is right.
* Wu vs SH ∇q magnitudes — diagnoses FD pole singularity in
  Step 09's `dq/dy` near 85.5°N.
"""

# %%
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
import config  # noqa: E402

WU = xr.open_dataset(Path(config.WU_OUT_DIR) / "pv_advection.nc")
SH = xr.open_dataset(Path(config.DATA_DIR) / "sh_out" / "sh_pv_advection.nc")

lats = WU.lat.values
lons = WU.lon.values
LON2D, LAT2D = np.meshgrid(lons, lats)

NPIECES = config.NPIECES
piece_names = [p["name"] for p in config.PIECES]
piece_hpa = [p["hPa"] for p in config.PIECES]


def wu_pvadv_from_inputs(ds):
    """Recompute Wu PVadv on the fly using Step 09's FD formulas.

    Done in-script (not by re-running Step 09) so the baseline NetCDF
    on GitHub stays untouched.
    """
    R_EARTH = 2.0e7 / np.pi
    DP = R_EARTH * np.radians(config.DLAT)
    DL = R_EARTH * np.radians(config.DLON)
    AP = np.cos(np.radians(lats))

    q = ds.Q_event_250.values
    q_fill = np.where(np.isnan(q), np.nanmean(q), q)
    NY = q.shape[0]
    dqdx = np.zeros_like(q_fill)
    dqdy = np.zeros_like(q_fill)
    dqdy[1:-1, :] = (q_fill[:-2, :] - q_fill[2:, :]) / (2.0 * DP)
    for i in range(1, NY - 1):
        dqdx[i, 1:-1] = (q_fill[i, 2:] - q_fill[i, :-2]) / (2.0 * DL * AP[i])

    U = ds.U_induced_250.values
    V = ds.V_induced_250.values
    PVadv = -(U * dqdx + V * dqdy) * 86400.0
    for ip in range(PVadv.shape[0]):
        PVadv[ip, 0, :] = PVadv[ip, -1, :] = np.nan
        PVadv[ip, :, 0] = PVadv[ip, :, -1] = np.nan
    return PVadv


if "PVadv" in WU.data_vars:
    WU_PVADV = WU.PVadv.values
else:
    print("Wu pv_advection.nc has no PVadv variable — recomputing inline.")
    WU_PVADV = wu_pvadv_from_inputs(WU)
SH_PVADV = SH.PVadv.values

# %%
def metrics(a, b, mask=None):
    """RMSE, pattern correlation, max|Δ| over mask (or all finite)."""
    if mask is None:
        mask = np.isfinite(a) & np.isfinite(b)
    else:
        mask = mask & np.isfinite(a) & np.isfinite(b)
    aa, bb = a[mask], b[mask]
    if aa.size == 0:
        return float("nan"), float("nan"), float("nan")
    rmse = float(np.sqrt(np.mean((aa - bb) ** 2)))
    if aa.std() < 1e-12 or bb.std() < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(aa, bb)[0, 1])
    mxd = float(np.max(np.abs(aa - bb)))
    return rmse, corr, mxd


# %%
print("=" * 80)
print(f"A/B compare: Wu (SOR+FD) vs SH (pvtend.sh_ops)  —  {config.EVENT_DATE}")
print("=" * 80)

print("\n>>> PV anomaly at 250 hPa (event − clim_mean)")
rmse, corr, mxd = metrics(WU.Q_anom_250.values, SH.Q_anom_250.values)
print(f"  full patch  : RMSE={rmse:6.3f} PVU  r={corr:+.3f}  max|Δ|={mxd:6.3f}")
m_extra = (LAT2D >= 20) & (LAT2D <= 70)
m_pole = LAT2D >= 70
rmse, corr, mxd = metrics(WU.Q_anom_250.values, SH.Q_anom_250.values, mask=m_extra)
print(f"  extratrop   : RMSE={rmse:6.3f} PVU  r={corr:+.3f}  max|Δ|={mxd:6.3f}")
rmse, corr, mxd = metrics(WU.Q_anom_250.values, SH.Q_anom_250.values, mask=m_pole)
print(f"  pole ≥70°N  : RMSE={rmse:6.3f} PVU  r={corr:+.3f}  max|Δ|={mxd:6.3f}")

print("\n>>> Induced wind speed at 250 hPa (per piece) — magnitudes only")
for ip, (name, hpa) in enumerate(zip(piece_names, piece_hpa)):
    su = np.sqrt(WU.U_induced_250.values[ip] ** 2 + WU.V_induced_250.values[ip] ** 2)
    ss = np.sqrt(SH.U_induced_250.values[ip] ** 2 + SH.V_induced_250.values[ip] ** 2)
    print(f"  piece {ip+1} {name:6s} ({hpa} hPa):")
    print(f"    Wu  |V|: mean={np.nanmean(su):6.2f} max={np.nanmax(su):6.2f} m/s")
    print(f"    SH  |V|: mean={np.nanmean(ss):6.2f} max={np.nanmax(ss):6.2f} m/s")

print("\n>>> PV advection per piece [PVU/day]")
for ip, (name, hpa) in enumerate(zip(piece_names, piece_hpa)):
    wu, sh = WU_PVADV[ip], SH_PVADV[ip]
    print(f"  piece {ip+1} {name:6s} ({hpa} hPa):")
    for label, mask in [("full ", None), ("extra", m_extra), ("pole ", m_pole)]:
        rmse, corr, mxd = metrics(wu, sh, mask=mask)
        print(f"    {label}: RMSE={rmse:6.2f}  r={corr:+.3f}  max|Δ|={mxd:6.1f}")

# %%
proj = ccrs.PlateCarree()
fig, axes = plt.subplots(2, NPIECES, figsize=(18, 11),
                         subplot_kw={"projection": proj})
fig.suptitle(
    f"A/B: Wu (top, SOR+FD)  vs  SH (bottom, pvtend.sh_ops)  —  "
    f"250-hPa PV advection per piece, {config.EVENT_DATE} 00Z",
    fontsize=12, fontweight="bold")

for ip in range(NPIECES):
    for row, (arr, label) in enumerate([(WU_PVADV, "Wu"), (SH_PVADV, "SH")]):
        ax = axes[row, ip]
        ax.set_extent([-170, -40, 10, 85], crs=proj)
        ax.add_feature(cfeature.COASTLINE, lw=0.5, edgecolor="gray")
        ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="lightgray")
        data = arr[ip]
        vmax = float(np.nanpercentile(np.abs(data), 98))
        if vmax < 1.0:
            vmax = 1.0
        cf = ax.pcolormesh(LON2D, LAT2D, data, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, transform=proj, shading="auto")
        plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02, label="PVU/day")
        ax.set_title(f"{label} — piece {ip+1} ({piece_hpa[ip]} hPa)\n"
                     f"[{np.nanmin(data):+.1f}, {np.nanmax(data):+.1f}] PVU/day",
                     fontsize=9)

fig.text(0.5, 0.01,
         "SH track caveat: pieces are vertical-mean ψ′, not 3-D piecewise "
         "PV inversion. Compare *patterns*, not absolute magnitudes.",
         ha="center", fontsize=8, style="italic", color="0.3")

out_png = Path(config.FIG_DIR) / "ab_wu_vs_sh.png"
out_pdf = Path(config.FIG_DIR) / "ab_wu_vs_sh.pdf"
fig.savefig(out_png, dpi=180, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"\n✓ Saved: {out_png}  ({out_png.stat().st_size/1e6:.1f} MB)")
print(f"✓ Saved: {out_pdf}")
plt.close(fig)

# %% Pole zoom — Wu rigid-lid artefact vs SH analytic pole
fig, axes = plt.subplots(2, NPIECES, figsize=(18, 10),
                         subplot_kw={"projection": ccrs.NorthPolarStereo(central_longitude=-105)})
fig.suptitle(
    f"Pole zoom (≥60°N): Wu (top) vs SH (bottom)  — checks for "
    f"rigid-lid noise leakage at 250 hPa",
    fontsize=12, fontweight="bold")
pc = ccrs.PlateCarree()
for ip in range(NPIECES):
    for row, (arr, label) in enumerate([(WU_PVADV, "Wu"), (SH_PVADV, "SH")]):
        ax = axes[row, ip]
        ax.set_extent([-180, 180, 60, 85.5], crs=pc)
        ax.add_feature(cfeature.COASTLINE, lw=0.5, edgecolor="gray")
        data = arr[ip]
        vmax = float(np.nanpercentile(np.abs(data), 98))
        if vmax < 1.0:
            vmax = 1.0
        cf = ax.pcolormesh(LON2D, LAT2D, data, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, transform=pc, shading="auto")
        plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.05, label="PVU/day")
        ax.set_title(f"{label} — piece {ip+1}", fontsize=9)
        ax.gridlines(lw=0.3, color="0.5")

out_png = Path(config.FIG_DIR) / "ab_pole_zoom.png"
fig.savefig(out_png, dpi=180, bbox_inches="tight")
print(f"✓ Saved: {out_png}")
plt.close(fig)
