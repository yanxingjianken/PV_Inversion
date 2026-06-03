#!/usr/bin/env python
"""Step 14 Pass C — Total PV inversion (BALNC) in Wu non-dim units.

Workflow:
    1. Load mean state (mean_state.nc) and event state (event_state.nc).
    2. q' = pv_event − pv_mean (anomaly drives the inversion).
    3. Non-dim H_bar, ψ_bar, q', θ_B, θ_T using Wu scales.
    4. Solve for balanced (H, ψ) via balnc_sh().
    5. Re-dim and write data/sh_ppvi/event_balanced.nc.

Usage:
    micromamba run -n blocking python steps/14_ppvi_pieces/run_pass_c.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import xarray as xr

from sh_ppvi.io import write_step_output
from sh_ppvi.balnc import balnc_sh
from sh_ppvi.nondim import (
    make_scales, nondim_H, nondim_psi, nondim_theta, nondim_q,
    nondim_thr, redim_H, redim_psi,
)

# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).resolve().parents[3] / "data"
MEAN_FILE       = DATA_DIR / "sh_ppvi" / "mean_state.nc"
EVENT_FILE      = DATA_DIR / "sh_ppvi" / "event_state.nc"
OUT_FILE        = DATA_DIR / "sh_ppvi" / "event_balanced.nc"
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"[Pass C] Loading mean state from {MEAN_FILE}")
    ds_mean  = xr.open_dataset(MEAN_FILE)
    pv_bar   = ds_mean["pv_bar"].values            # (NW, nlat, nlon)
    H_bar    = ds_mean["H_bar"].values
    psi_bar  = ds_mean["psi_bar"].values
    theta_B_bar = ds_mean["theta_B_bar"].values    # (nlat, nlon)
    theta_T_bar = ds_mean["theta_T_bar"].values
    lat      = ds_mean["latitude"].values
    lon      = ds_mean["longitude"].values
    plev_hpa = ds_mean["level"].values if "level" in ds_mean else None

    print(f"[Pass C] Loading event state from {EVENT_FILE}")
    ds_ev   = xr.open_dataset(EVENT_FILE)
    pv_ev   = ds_ev["pv"].values
    H_ev    = ds_ev["H"].values
    psi_ev  = ds_ev["psi"].values
    theta_B = ds_ev["theta_B"].values
    theta_T = ds_ev["theta_T"].values

    q_prime = pv_ev - pv_bar
    print(f"  q' range: {q_prime.min():.4g} .. {q_prime.max():.4g}")
    print(f"  q'  RMS : {np.sqrt(np.mean(q_prime**2)):.4g}")

    # ── Build Wu scales for this grid ──────────────────────────────────────
    dlon = float(lon[1] - lon[0])
    sc   = make_scales(dlon_deg=dlon)
    print(f"[Pass C] {sc}")

    # ── Non-dimensionalise inputs ──────────────────────────────────────────
    # Use event state as the initial guess — matching Wu's Fortran which reads
    # H and S from event files (qinvert21_94.f L221–L300).  Starting from the
    # mean state places (H_init, psi_init) far from the balanced event field;
    # the ZL bilinear term then has gain ~6 per Picard step → divergence.
    # Event-state init ≈ balanced event field, so ZL_frozen ≈ ZL_balanced and
    # the frozen-ZL Picard converges accurately.
    q_total      = pv_bar + q_prime                   # ≡ pv_ev, but explicit
    H_init_nd    = nondim_H(H_ev, sc)
    psi_init_nd  = nondim_psi(psi_ev, sc)
    q_nd         = nondim_q(q_total, sc)
    thB_nd    = nondim_theta(theta_B, sc)
    thT_nd    = nondim_theta(theta_T, sc)
    thr_nd    = nondim_thr(1.0, sc)                # 1 m geopotential height ≈

    print(f"  H_nd  range:  {H_init_nd.min():.3g} .. {H_init_nd.max():.3g}")
    print(f"  ψ_nd  range:  {psi_init_nd.min():.3g} .. {psi_init_nd.max():.3g}")
    print(f"  q_nd  range:  {q_nd.min():.3g} .. {q_nd.max():.3g}")
    print(f"  θB_nd range:  {thB_nd.min():.3g} .. {thB_nd.max():.3g}")
    print(f"  thr_nd     :  {thr_nd:.4g}")

    print("[Pass C] Running BALNC SOR inversion ...")
    result = balnc_sh(
        H_init    = H_init_nd,
        psi_init  = psi_init_nd,
        q         = q_nd,
        theta_B   = thB_nd,
        theta_T   = thT_nd,
        lat       = lat,
        lon       = lon,
        scales    = sc,
        max_total = 200,
        thr       = thr_nd,
        part      = 0.05,
        verbose   = True,
    )

    H_bal_nd   = result["H"]
    psi_bal_nd = result["psi"]
    hist       = result["hist"]
    n_outer    = len(hist)
    last       = hist[-1]
    print(f"  Outer iters used: {n_outer}; last: Δψ={last['dpsi']:.3g}, ΔH={last['dH']:.3g}")

    # ── Re-dimensionalise ──────────────────────────────────────────────────
    H_bal   = redim_H(H_bal_nd, sc)
    psi_bal = redim_psi(psi_bal_nd, sc)
    print(f"  H_bal  [m²/s²] range: {H_bal.min():.4g} .. {H_bal.max():.4g}")
    print(f"  ψ_bal  [m²/s]  range: {psi_bal.min():.4g} .. {psi_bal.max():.4g}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_step_output(
        str(OUT_FILE),
        lat, lon,
        plev_hpa = plev_hpa,
        H_bal    = H_bal,
        psi_bal  = psi_bal,
        q_prime  = q_prime,
        pv_ev    = pv_ev,
        pv_bar   = pv_bar,
        H_bar    = H_bar,
        psi_bar  = psi_bar,
    )
    print(f"[Pass C] Written → {OUT_FILE}")


if __name__ == "__main__":
    main()
