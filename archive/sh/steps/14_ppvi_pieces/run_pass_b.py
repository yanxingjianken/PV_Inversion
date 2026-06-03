#!/usr/bin/env python
"""Step 14 Pass B — Event-state PV computation.

Mirrors pvpialln_94UV.f applied to the event ERA5 fields.

Workflow:
    1. Load ERA5 event file.
    2. Compute θ, ζ, ψ, Ertel PV via pvpialln_sh().
    3. Write to data/sh_ppvi/event_state.nc.

Usage:
    micromamba run -n blocking python steps/14_ppvi_pieces/run_pass_b.py
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from sh_ppvi.io import load_global_era5, prep_event, write_step_output
from sh_ppvi.pv_calc import pvpialln_sh

# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parents[3] / "data"
ERA5_FILE  = DATA_DIR / "era5" / "era5_2025-01-08_00Z.nc"
OUT_FILE   = DATA_DIR / "sh_ppvi" / "event_state.nc"
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"[Pass B] Loading ERA5 from {ERA5_FILE}")
    ds  = load_global_era5(str(ERA5_FILE))
    ev  = prep_event(ds, hemisphere="NH")

    u, v, T   = ev["u"], ev["v"], ev["T"]
    H, lat, lon = ev["Phi"], ev["lat"], ev["lon"]

    print(f"[Pass B] Computing event PV (pvpialln_sh) ...")
    out = pvpialln_sh(H, T, u, v, lat, lon)

    print(f"  PV range: {out['pv'].min():.4g} .. {out['pv'].max():.4g}")
    print(f"  ψ  range: {out['psi'].min():.4g} .. {out['psi'].max():.4g}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_step_output(
        str(OUT_FILE),
        lat, lon,
        plev_hpa=ev["plev_hpa"],
        theta     = out["theta"],
        psi       = out["psi"],
        H         = H,
        zeta      = out["zeta"],
        pv        = out["pv"],
        dth_dx    = out["dth_dx"],
        dth_dy    = out["dth_dy"],
        dth_dpi   = out["dth_dpi"],
        du_dpi    = out["du_dpi"],
        dv_dpi    = out["dv_dpi"],
        theta_B   = out["theta_B"],
        theta_T   = out["theta_T"],
        u         = u,
        v         = v,
    )
    print(f"[Pass B] Written → {OUT_FILE}")


if __name__ == "__main__":
    main()
