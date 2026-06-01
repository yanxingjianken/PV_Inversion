"""Step 15: Davis Fig 8 replica using SH-PPVI piecewise winds.

Reads data/sh_ppvi/piecewise.nc and produces a 3-panel map
(lower / middle / upper PV pieces) of the V-component induced
at 250 hPa by each piece, overlaid on the event Z500 field.

Each panel uses its own colour-bar range, auto-scaled to the p98 of
that panel and snapped to a nice grid (mirrors the Wu Fig-8 logic so
the two figures are directly comparable).

Saved to fig8_sh_ppvi.{png,pdf} alongside this script.

Usage:
    micromamba run -n blocking python sh/steps/15_fig8_sh_ppvi/fig8.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import numpy as np
from config import (
    DATA_DIR, ERA5_DIR, PLEVS as PLEVS_HPA,
    EVENT_DATE, PIECES, LON_W, LON_E, LAT_S, LAT_N,
)

# 0-based index of 250 hPa
K250 = int(np.where(np.isclose(PLEVS_HPA, 250))[0][0])  # typically 8

PIECE_FILE = Path(DATA_DIR) / "sh_ppvi" / "piecewise.nc"
ERA5_FILE  = Path(ERA5_DIR) / f"era5_{EVENT_DATE:%Y-%m-%d}_00Z.nc"
OUT_BASE   = Path(__file__).resolve().parent / "fig8_sh_ppvi"

TITLES = {
    "lower" : "(a) Lower trop. (1000–925 hPa)",
    "middle": "(b) Middle trop. (850–700 hPa)",
    "upper" : "(c) Upper trop. (600–250 hPa)",
}
CLEV_Z = np.arange(4800, 5800, 80)   # gpm for background Z500
_NICE  = np.array([0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50])


def _nice_vmax(field: np.ndarray) -> float:
    """Snap p98(|field|) to nearest value in _NICE."""
    v = float(np.nanpercentile(np.abs(field), 98))
    if v < 0.5:
        v = 0.5
    return float(_NICE[np.argmin(np.abs(_NICE - v))])


def main():
    # ── Load piecewise outputs ──────────────────────────────────────
    print(f"[Step 15] Loading {PIECE_FILE}")
    ds = xr.open_dataset(PIECE_FILE)
    lat = ds.latitude.values
    lon = ds.longitude.values

    # ── Load ERA5 Z500 background ───────────────────────────────────
    ds_ev = xr.open_dataset(ERA5_FILE)
    z_all = ds_ev["z"].values.squeeze()[:, ::-1, :]   # (10, 61, 240), ascending lat
    idx500 = list(PLEVS_HPA).index(500)
    z500 = z_all[idx500] / 9.81                        # convert Φ→gpm

    # ── Figure ──────────────────────────────────────────────────────
    proj   = ccrs.PlateCarree()
    extent = [LON_W, LON_E, LAT_S, LAT_N]

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )
    fig.suptitle(
        f"SH-PPVI induced meridional wind at 250 hPa — {EVENT_DATE:%d %b %Y}",
        fontsize=13,
    )

    # crop indices once (same for every panel)
    lon_w360 = LON_W % 360
    lon_e360 = LON_E % 360
    if lon_w360 > lon_e360:
        ilon = np.where((lon >= lon_w360) | (lon <= lon_e360))[0]
    else:
        ilon = np.where((lon >= lon_w360) & (lon <= lon_e360))[0]
    ilat = np.where((lat >= LAT_S) & (lat <= LAT_N))[0]
    lon_p = lon[ilon]; lat_p = lat[ilat]
    z_p   = z500[np.ix_(ilat, ilon)]

    panel_vmax = []
    for ax, piece in zip(axes, PIECES):
        name = piece["name"]
        v_250 = ds[f"v_{name}"].values[K250]           # (nlat, nlon)
        v_p   = v_250[np.ix_(ilat, ilon)]

        # ── per-panel cbar range (p98, snapped to nice grid) ────────
        vmax = _nice_vmax(v_p)
        panel_vmax.append(vmax)
        # 11 contour levels symmetric around 0
        levels = np.linspace(-vmax, vmax, 21)

        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle=":")
        ax.add_feature(cfeature.LAND,      facecolor="whitesmoke", zorder=0)

        # Z500 background contours
        cz = ax.contour(lon_p, lat_p, z_p, levels=CLEV_Z,
                        colors="gray", linewidths=0.8, transform=proj)
        ax.clabel(cz, fmt="%d", fontsize=6)

        # V_250 filled contours (per-panel scale)
        cf = ax.contourf(lon_p, lat_p, v_p, levels=levels,
                         cmap="RdBu_r", extend="both", transform=proj)
        cb = plt.colorbar(cf, ax=ax, orientation="horizontal",
                          pad=0.03, shrink=0.9, aspect=30, fraction=0.04)
        cb.set_label(f"V$_{{250}}$  (m/s)   [\u00b1{vmax:g}]", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                          alpha=0.6, linestyle="--")
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.MultipleLocator(20)
        gl.ylocator = mticker.MultipleLocator(20)
        gl.xlabel_style = gl.ylabel_style = {"size": 7}

        ax.set_title(TITLES[name], fontsize=10)

    print(f"  Per-panel V_250 vmax (m/s): {panel_vmax}")

    # ── Save ────────────────────────────────────────────────────────
    for ext in ("png", "pdf"):
        out = f"{OUT_BASE}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out}")

    plt.close(fig)
    print("[Step 15] Done.")


if __name__ == "__main__":
    main()
