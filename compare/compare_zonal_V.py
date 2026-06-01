"""compare/compare_zonal_V.py — Zonal-mean profile of |V_250| per piece.

Shows latitude dependence of induced V_250 magnitude for Wu vs SH-PPVI on
the Wu patch (10.5–85.5°N).  Three rows = three vertical pieces.

Output: data/figs/compare_zonal_V.{png,pdf}.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from config import DATA_DIR, PLEVS as PLEVS_HPA, EVENT_DATE

WU_FILE  = Path(DATA_DIR) / "wu_out"  / "pv_advection.nc"
SH_FILE  = Path(DATA_DIR) / "sh_ppvi" / "piecewise.nc"
OUT_BASE = Path(DATA_DIR) / "figs"    / "compare_zonal_V"

K250   = int(np.where(np.isclose(PLEVS_HPA, 250))[0][0])
PIECES = ["lower", "middle", "upper"]
TITLES = ["(a) Lower (1000–925 hPa)", "(b) Middle (850–700 hPa)",
          "(c) Upper (600–250 hPa)"]


def main():
    ds_wu  = xr.open_dataset(WU_FILE)
    wu_lat = ds_wu["lat"].values
    wu_lon = ds_wu["lon"].values
    Vwu    = ds_wu["V_induced_250"].values   # (3, 51, 87)

    ds_sh  = xr.open_dataset(SH_FILE)
    sh_lat = ds_sh["latitude"].values
    sh_lon = ds_sh["longitude"].values
    ilat_sh = [int(np.argmin(np.abs(sh_lat - lat))) for lat in wu_lat]
    ilon_sh = [int(np.argmin(np.abs(sh_lon - lon))) for lon in wu_lon]
    Vsh = np.stack([
        ds_sh[f"v_{name}"].values[K250][np.ix_(ilat_sh, ilon_sh)]
        for name in PIECES
    ], axis=0)

    # Zonal-mean RMS along longitude axis
    rms_wu = np.sqrt(np.nanmean(Vwu ** 2, axis=2))   # (3, 51)
    rms_sh = np.sqrt(np.nanmean(Vsh ** 2, axis=2))

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True,
                              constrained_layout=True)
    fig.suptitle(
        f"Zonal RMS of induced V$_{{250}}$ vs latitude — {EVENT_DATE:%d %b %Y}\n"
        f"Wu (FD-SOR, log scale)  vs  SH-PPVI (spectral, log scale)",
        fontsize=11,
    )
    for ax, name, title in zip(axes, PIECES, TITLES):
        ir = PIECES.index(name)
        ax.semilogy(wu_lat, rms_wu[ir], "k-",  lw=2, label="Wu  FD-SOR")
        ax.semilogy(wu_lat, rms_sh[ir], "r--", lw=2, label="SH-PPVI spectral")
        ax.set_ylabel("|V$_{250}$|  RMS (m/s)")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlim(wu_lat.max(), wu_lat.min())   # high lat → low lat (left→right)
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.axvspan(80, 90, color="lightblue", alpha=0.3,
                   label="polar zone (Wu LBC at 85.5°N)")
        ax.legend(loc="lower left", fontsize=8)
    axes[-1].set_xlabel("Latitude (°N)")

    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_BASE}.{ext}", dpi=130, bbox_inches="tight")
        print(f"  Saved → {OUT_BASE}.{ext}")
    plt.close(fig)

    # ── Print mid-lat vs polar ratio ────────────────────────────────
    print("\n=== Zonal RMS at selected latitudes (m/s) ===")
    print(f"  {'lat':>6s} | {'piece':<8s} | {'Wu':>10s} | {'SH':>10s} | "
          f"{'SH/Wu':>7s}")
    for lat_target in [85.5, 70.5, 45.0, 25.5, 10.5]:
        i = int(np.argmin(np.abs(wu_lat - lat_target)))
        for ir, name in enumerate(PIECES):
            r = rms_sh[ir, i] / max(rms_wu[ir, i], 1e-9)
            print(f"  {wu_lat[i]:6.1f} | {name:<8s} | "
                  f"{rms_wu[ir,i]:10.4f} | {rms_sh[ir,i]:10.4f} | {r:7.3f}")


if __name__ == "__main__":
    main()
