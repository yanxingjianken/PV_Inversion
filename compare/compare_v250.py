"""compare/compare_v250.py — Wu vs SH-PPVI: induced V_250 maps.

3×3 panel grid:
  rows = vertical pieces (lower / middle / upper)
  cols = [Wu finite-difference SOR] | [SH-PPVI spectral-vertical] | [SH − Wu]

Each row uses its own colour-bar range (p98 of the Wu+SH panels, snapped
to a nice grid). The diff column uses ±max(|SH−Wu|).

Both methods are sampled on the Wu domain (lat 10.5–85.5°N, lon 190.5–319.5°E,
51×87 at 1.5° spacing). The SH output is also extracted from its full-NH grid
on the same indices, so the comparison is point-wise.

Saved to data/figs/compare_v250.{png,pdf}.

Usage:
    micromamba run -n blocking python compare/compare_v250.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from config import (
    DATA_DIR, PLEVS as PLEVS_HPA,
    EVENT_DATE, LON_W, LON_E, LAT_S, LAT_N,
)

WU_FILE  = Path(DATA_DIR) / "wu_out"  / "pv_advection.nc"
SH_FILE  = Path(DATA_DIR) / "sh_ppvi" / "piecewise.nc"
OUT_BASE = Path(DATA_DIR) / "figs"    / "compare_v250"

K250    = int(np.where(np.isclose(PLEVS_HPA, 250))[0][0])
PIECES  = ["lower", "middle", "upper"]
TITLES  = ["1000–925 hPa", "850–700 hPa", "600–250 hPa"]
_NICE   = np.array([0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50])


def _nice_vmax(a):
    v = float(np.nanpercentile(np.abs(a), 98))
    if v < 0.5:
        v = 0.5
    return float(_NICE[np.argmin(np.abs(_NICE - v))])


def main():
    # ── Wu (already on the patch grid) ──────────────────────────────
    ds_wu  = xr.open_dataset(WU_FILE)
    wu_lat = ds_wu["lat"].values    # 85.5 → 10.5 descending
    wu_lon = ds_wu["lon"].values    # 190.5 → 319.5
    Vwu    = ds_wu["V_induced_250"].values   # (3, 51, 87)

    # ── SH-PPVI on full NH; subset to Wu grid ───────────────────────
    ds_sh  = xr.open_dataset(SH_FILE)
    sh_lat = ds_sh["latitude"].values   # 0 → 90 ascending
    sh_lon = ds_sh["longitude"].values  # 0 → 358.5

    # Index map: Wu lat descending → ascending order
    ilat_sh = [int(np.argmin(np.abs(sh_lat - lat))) for lat in wu_lat]
    ilon_sh = [int(np.argmin(np.abs(sh_lon - lon))) for lon in wu_lon]
    Vsh = np.stack([
        ds_sh[f"v_{name}"].values[K250][np.ix_(ilat_sh, ilon_sh)]
        for name in PIECES
    ], axis=0)   # (3, 51, 87)

    # ── Figure ──────────────────────────────────────────────────────
    proj   = ccrs.PlateCarree()
    extent = [LON_W, LON_E, LAT_S, LAT_N]

    fig, axes = plt.subplots(
        3, 3, figsize=(15, 13),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )
    fig.suptitle(
        f"V$_{{250}}$ induced by piecewise PV inversion — {EVENT_DATE:%d %b %Y}\n"
        f"columns: Wu (FD-SOR)  |  SH-PPVI (spectral)  |  SH − Wu",
        fontsize=12,
    )

    panel_summary = []
    for ir, (name, title) in enumerate(zip(PIECES, TITLES)):
        # Row-wise per-method vmax
        vmax_row  = _nice_vmax(np.concatenate([Vwu[ir].ravel(),
                                               Vsh[ir].ravel()]))
        diff       = Vsh[ir] - Vwu[ir]
        vmax_diff = _nice_vmax(diff)

        rms_wu  = float(np.sqrt(np.nanmean(Vwu[ir] ** 2)))
        rms_sh  = float(np.sqrt(np.nanmean(Vsh[ir] ** 2)))
        rmse    = float(np.sqrt(np.nanmean(diff ** 2)))
        panel_summary.append((name, rms_wu, rms_sh, rmse))

        for ic, (data, vmax, label) in enumerate([
            (Vwu[ir], vmax_row,  f"Wu  RMS={rms_wu:.2f}"),
            (Vsh[ir], vmax_row,  f"SH  RMS={rms_sh:.2f}"),
            (diff,    vmax_diff, f"SH−Wu  RMSE={rmse:.2f}"),
        ]):
            ax = axes[ir, ic]
            ax.set_extent(extent, crs=proj)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle=":")
            levels = np.linspace(-vmax, vmax, 21)
            cf = ax.contourf(wu_lon, wu_lat, data, levels=levels,
                             cmap="RdBu_r", extend="both", transform=proj)
            cb = plt.colorbar(cf, ax=ax, orientation="horizontal",
                              pad=0.04, shrink=0.85, aspect=25, fraction=0.04)
            cb.set_label(f"V$_{{250}}$ (m/s)  [\u00b1{vmax:g}]", fontsize=7)
            cb.ax.tick_params(labelsize=6)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                              color="gray", alpha=0.5, linestyle="--")
            gl.top_labels = gl.right_labels = False
            gl.xlocator = mticker.MultipleLocator(30)
            gl.ylocator = mticker.MultipleLocator(20)
            gl.xlabel_style = gl.ylabel_style = {"size": 6}
            ax.set_title(f"{title}\n{label}", fontsize=9)

    # ── Save & print summary ────────────────────────────────────────
    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_BASE}.{ext}", dpi=130, bbox_inches="tight")
        print(f"  Saved → {OUT_BASE}.{ext}")
    plt.close(fig)

    print("\n=== Wu vs SH RMS summary (V_250, m/s) — Wu patch ===")
    print(f"  {'piece':<8s}  {'Wu RMS':>8s}  {'SH RMS':>8s}  "
          f"{'SH−Wu RMSE':>11s}  {'rel %':>6s}")
    for name, rwu, rsh, e in panel_summary:
        rel = e / max(rwu, 1e-6) * 100
        print(f"  {name:<8s}  {rwu:8.3f}  {rsh:8.3f}  {e:11.3f}  {rel:6.1f}")


if __name__ == "__main__":
    main()
