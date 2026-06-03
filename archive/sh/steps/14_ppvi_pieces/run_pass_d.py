#!/usr/bin/env python
"""Step 14 Pass D — Piecewise perturbation PV inversion (BALP).

Mirrors qinvertp21_94.f SUBROUTINE BALP.

Workflow:
    1. Load mean state and event state.
    2. Compute perturbation PV q' = q_event − q_mean.
    3. For each piece (lower/mid/upper), call balp_sh().
    4. Write results for all pieces + sum to data/sh_ppvi/event_pieces.nc.

Usage:
    micromamba run -n blocking python steps/14_ppvi_pieces/run_pass_d.py
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import xarray as xr

from sh_ppvi.io import write_step_output
from sh_ppvi.balp import balp_sh, make_pieces

# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR  = Path(__file__).resolve().parents[3] / "data"
MEAN_FILE = DATA_DIR / "sh_ppvi" / "mean_state.nc"
EVENT_FILE= DATA_DIR / "sh_ppvi" / "event_state.nc"
OUT_FILE  = DATA_DIR / "sh_ppvi" / "event_pieces.nc"
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"[Pass D] Loading mean state from {MEAN_FILE}")
    ds_mean  = xr.open_dataset(MEAN_FILE)
    pv_bar   = ds_mean["pv_bar"].values
    H_bar    = ds_mean["H_bar"].values
    psi_bar  = ds_mean["psi_bar"].values
    lat      = ds_mean["latitude"].values
    lon      = ds_mean["longitude"].values
    plev_hpa = ds_mean["level"].values if "level" in ds_mean else None

    print(f"[Pass D] Loading event state from {EVENT_FILE}")
    ds_ev   = xr.open_dataset(EVENT_FILE)
    pv_ev   = ds_ev["pv"].values
    theta_B = ds_ev["theta_B"].values
    theta_T = ds_ev["theta_T"].values

    # Perturbation θ boundaries
    theta_B_bar = ds_mean.get("theta_B_bar", xr.DataArray(np.zeros_like(theta_B))).values
    theta_T_bar = ds_mean.get("theta_T_bar", xr.DataArray(np.zeros_like(theta_T))).values
    q_prime = pv_ev - pv_bar
    dtheta_B = theta_B - theta_B_bar
    dtheta_T = theta_T - theta_T_bar

    print(f"  q' range: {q_prime.min():.4g} .. {q_prime.max():.4g}")

    mean_state = {"H": H_bar, "psi": psi_bar}
    pieces     = make_pieces()

    print(f"[Pass D] Running BALP for {len(pieces)} pieces ...")
    result = balp_sh(
        mean_state    = mean_state,
        event_q       = q_prime,
        event_theta_B = dtheta_B,
        event_theta_T = dtheta_T,
        lat           = lat,
        lon           = lon,
        pieces        = pieces,
        omegs         = 1.4,
        omegh         = 1.4,
        max_it        = 200,
        max_total     = 200,
        thr           = 0.01,
        part          = 0.5,
    )

    psi_sum = result["psi_sum"]
    H_sum   = result["H_sum"]
    for i, piece in enumerate(pieces):
        ph = result["hist"][i]
        last = ph["hist"][-1] if ph["hist"] else {}
        print(f"  Piece {piece['name']}: {len(ph['hist'])} outer iters; last={last}")

    # Build output arrays
    out_arrays = {
        "psi_sum":  psi_sum,
        "H_sum":    H_sum,
        "q_prime":  q_prime,
    }
    for i, piece in enumerate(pieces):
        name = piece["name"]
        out_arrays[f"psi_{name}"] = result["psi_pieces"][i]
        out_arrays[f"H_{name}"]   = result["H_pieces"][i]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_step_output(str(OUT_FILE), lat, lon, plev_hpa=plev_hpa, **out_arrays)
    print(f"[Pass D] Written → {OUT_FILE}")

    # Closure check: Σψ_pieces ≈ total δψ within 5%
    psi_event = ds_ev["psi"].values if "psi" in ds_ev else None
    psi_bar_v = psi_bar
    if psi_event is not None:
        dpsi = psi_event - psi_bar_v
        ratio = np.mean(np.abs(psi_sum)) / (np.mean(np.abs(dpsi)) + 1e-30)
        print(f"  Closure check: Σψ_pieces / δψ = {ratio:.3f}  (target ≈ 1.0 ± 5%)")


if __name__ == "__main__":
    main()
