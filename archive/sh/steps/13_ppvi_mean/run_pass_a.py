#!/usr/bin/env python
"""Step 13 Pass A — Mean-state PV computation from 30-day climatology.

Builds a 30-day running mean (±15 days) centred on EVENT_DATE, then computes
θ, ζ, ψ, Ertel PV via pvpialln_sh() on that mean state.

Output → data/sh_ppvi/mean_state.nc.

Usage:
    micromamba run -n blocking python steps/13_ppvi_mean/run_pass_a.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import xarray as xr

from sh_ppvi.io import prep_event, write_step_output
from sh_ppvi.pv_calc import pvpialln_sh
from sh_ppvi.clim import list_clim_files, load_clim_mean

# ─────────────────────────────────────────────────────────────────────────────
EVENT_DATE   = "2025-01-08"
HALF_WINDOW  = 15                     # ±15 days → 31-day window
DATA_DIR     = Path(__file__).resolve().parents[3] / "data"
ERA5_DIR     = DATA_DIR / "era5"
OUT_FILE     = DATA_DIR / "sh_ppvi" / "mean_state.nc"
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    files = list_clim_files(ERA5_DIR, EVENT_DATE, half_window=HALF_WINDOW)
    print(f"[Pass A] {len(files)} files in ±{HALF_WINDOW}-day window around {EVENT_DATE}")
    print(f"         {files[0].name} ... {files[-1].name}")

    ds_mean = load_clim_mean(files)
    ev      = prep_event(ds_mean, hemisphere="NH")

    u, v, T   = ev["u"], ev["v"], ev["T"]
    H, lat, lon = ev["Phi"], ev["lat"], ev["lon"]
    print(f"  domain: lat {lat[0]:.1f}..{lat[-1]:.1f}, lon {lon[0]:.1f}..{lon[-1]:.1f}")
    print(f"  shape:  {u.shape}")

    print("[Pass A] Computing climatological PV (pvpialln_sh) ...")
    out = pvpialln_sh(H, T, u, v, lat, lon)
    print(f"  PV range:  {out['pv'].min():.4g} .. {out['pv'].max():.4g}")
    print(f"  ψ  range:  {out['psi'].min():.4g} .. {out['psi'].max():.4g}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_step_output(
        str(OUT_FILE),
        lat, lon,
        plev_hpa=ev["plev_hpa"],
        theta_bar   = out["theta"],
        psi_bar     = out["psi"],
        H_bar       = H,
        zeta_bar    = out["zeta"],
        pv_bar      = out["pv"],
        dth_dx_bar  = out["dth_dx"],
        dth_dy_bar  = out["dth_dy"],
        dth_dpi_bar = out["dth_dpi"],
        du_dpi_bar  = out["du_dpi"],
        dv_dpi_bar  = out["dv_dpi"],
        theta_B_bar = out["theta_B"],
        theta_T_bar = out["theta_T"],
        u_bar       = u,
        v_bar       = v,
    )
    print(f"[Pass A] Written → {OUT_FILE}  (mean of {len(files)} days)")


if __name__ == "__main__":
    main()
