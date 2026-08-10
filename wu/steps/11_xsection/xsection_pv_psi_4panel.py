# %% [markdown]
# Step 11 — 2D Longitude–Pressure (x–p) Cross-Sections: PV anomaly + 3 piece ψ′
#
# Four flat 2D panels, each a longitude–pressure cross-section at the blocking
# latitude (the Z500 block centre), shaded with filled contours:
#   (a) total PV anomaly  q' = q_event − q_mean        [PVU]
#   (b) ψ' induced by the SURFACE piece (1000 hPa boundary θ)
#   (c) ψ' induced by the LOWER piece   (850–500 hPa PV)
#   (d) ψ' induced by the UPPER piece   (400–200 hPa PV + 100 hPa top θ)
#
# Reads: piecewise_psi.nc (psi[piece,plev,lat,lon]) + mean_clim.nc / event.nc.
# %%
import numpy as np, xarray as xr, matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path
import sys

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
import config

STEP_DIR = Path(__file__).resolve().parent
OUT_DIR = Path(config.WU_OUT_DIR)
CLIM_DIR = Path(config.CLIM_DIR)

# ── Load piecewise ψ′ (SI m² s⁻¹) ──────────────────────────────────────────────
ds = xr.open_dataset(OUT_DIR / "piecewise_psi.nc")
PSI = ds["psi"].values            # (piece, plev, lat, lon) — surface, lower, upper
plevs = ds["plev"].values         # hPa, 1000 … 100
lats = ds["lat"].values
lons = ds["lon"].values
piece_hpa = {p["name"]: p["hPa"] for p in config.PIECES}

# ── Total PV anomaly q' (SI → PVU) ─────────────────────────────────────────────
ds_m = xr.open_dataset(CLIM_DIR / "mean_clim.nc")
ds_e = xr.open_dataset(CLIM_DIR / "event.nc")
P0, Rd, Cp, RE, OM, g = 1.0e5, 287.0, 1004.0, 6.371e6, 7.292e-5, 9.81
p_pa = ds_m.pressure_level.values.astype(float) * 100.0


def ertel_pv(t, u, v):
    th = t * (P0 / p_pa[:, None, None]) ** (Rd / Cp)
    dthdp = np.gradient(th, p_pa, axis=0)
    dla = np.deg2rad(np.gradient(lats))[None, :, None]
    dlo = np.deg2rad(np.gradient(lons))[None, None, :]
    cl = np.cos(np.deg2rad(lats))[None, :, None]
    zeta = np.gradient(v, axis=2) / (RE * cl * dlo) - np.gradient(u, axis=1) / (RE * dla)
    f = 2 * OM * np.sin(np.deg2rad(lats))[None, :, None]
    return -g * (zeta + f) * dthdp


PV_anom = (ertel_pv(ds_e["t"].values, ds_e["u"].values, ds_e["v"].values)
           - ertel_pv(ds_m["t"].values, ds_m["u"].values, ds_m["v"].values)) * 1e6  # PVU

# ── Blocking latitude/longitude from the event Z500 (NE-Pacific / W-NA) ─────────
z500 = ds_e["z"].values[int(np.argmin(np.abs(ds_e.pressure_level.values - 500)))] / 9.81
_lo360 = lons % 360
reg = (lats[:, None] >= 25) & (lats[:, None] <= 65) & (_lo360[None, :] >= 170) & (_lo360[None, :] <= 260)
lat_idx, lon_idx = np.unravel_index(np.nanargmax(np.where(reg, z500, -np.inf)), z500.shape)
LAT0, LON0 = float(lats[lat_idx]), float(lons[lon_idx])
print(f"Block centre: lat={LAT0:.1f}°N, lon={LON0:.1f}°E (={360-LON0:.1f}°W); "
      f"cross-sections taken along {LAT0:.1f}°N.")

# ── Plot: 2x2 longitude–pressure cross-sections at the block latitude ──────────
LON2D, PLEV2D = np.meshgrid(lons, plevs)
PSI_DISP = PSI / 1e6  # 10⁶ m² s⁻¹

specs = [
    (PV_anom[:, lat_idx, :],     "(a) PV anomaly  q′",                              "PVU"),
    (PSI_DISP[0, :, lat_idx, :], f"(b) ψ′ from SURFACE piece ({piece_hpa['surface']})", "10⁶ m² s⁻¹"),
    (PSI_DISP[1, :, lat_idx, :], f"(c) ψ′ from LOWER piece ({piece_hpa['lower']})",     "10⁶ m² s⁻¹"),
    (PSI_DISP[2, :, lat_idx, :], f"(d) ψ′ from UPPER piece ({piece_hpa['upper']})",     "10⁶ m² s⁻¹"),
]
YT = [1000, 850, 700, 500, 400, 300, 250, 200, 100]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, (data2d, title, unit) in zip(axes.flat, specs):
    vmax = float(np.nanpercentile(np.abs(data2d), 99)) or 1.0
    levels = np.linspace(-vmax, vmax, 21)
    cf = ax.contourf(LON2D, PLEV2D, data2d, levels=levels, cmap="RdBu_r", extend="both")
    cb = plt.colorbar(cf, ax=ax, pad=0.02, shrink=0.9)
    cb.set_label(unit, fontsize=9)
    ax.axvline(LON0, color="k", lw=1.0, ls="--", alpha=0.6)  # block longitude
    ax.set_yscale("log")
    ax.set_ylim(1000, 100)
    ax.set_yticks(YT)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticklabels([str(p) for p in YT], fontsize=8)
    ax.minorticks_off()
    ax.set_xlim(lons.min(), lons.max())
    ax.set_xlabel("Longitude [°E]", fontsize=9)
    ax.set_ylabel("Pressure [hPa]", fontsize=9)
    ax.set_title(title + "   (blue = cyclonic ψ′)" * (title[1] != "a"),
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25)

fig.suptitle(
    "Longitude–Pressure Cross-Sections of the Piecewise PV Inversion\n"
    f"CA Blocking 2025-01-08 00Z  |  along the block latitude {LAT0:.1f}°N "
    f"(dashed = block longitude {LON0:.0f}°E)  |  9 levels, INLIN=1",
    fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])

png = STEP_DIR / "xsection_pv_psi_4panel.png"
fig.savefig(png, dpi=160, bbox_inches="tight")
fig.savefig(STEP_DIR / "xsection_pv_psi_4panel.pdf", bbox_inches="tight")
print(f"✓ Saved: {png}")
