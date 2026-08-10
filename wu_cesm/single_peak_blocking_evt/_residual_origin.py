#!/usr/bin/env python3
"""Diagnostic (not a figure): what IS the residual in 03 panel (d)?

03(d) = (V_block + V_eddy) - V_rot = V_upper - V_rot. It is 27% of the reference, which is not
small, and the claim in 03's caption is that this is the LOWER + SURFACE contribution rather than
an error. That claim is testable: invert those pieces too and see how much of the residual they
account for.

  lower   = Wu levels {2,3,4} = 850/700/500 interior PV, no theta
  surface = the BOTTOM boundary theta at 1000 hPa (Wu level 1), no interior PV

If V_lower + V_surface reproduces -(d), the decomposition is complete and 03(d) is physics.
Whatever is left over after that is the part nonlinear balance cannot represent — the unbalanced
rotational flow — plus solver error. That leftover is a FLOOR: no amount of extra iteration or
extra pieces removes it, because `V_rot` is the full Helmholtz rotational wind and the inversion
only ever produces a balanced one.

Run:  micromamba run -n blocking python _residual_origin.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import _evt
from wu_pv import wu_pv_interior
import ppvi_block_wind as PB
from pvtend.ppvi.solver import PassDParams, _from_core, NL
from pvtend.ppvi.winds import psi_to_winds

HERE = Path(__file__).resolve().parent
GRID = _evt.GRID


def invert_pieces(e, thrsh=1e-3):
    """psi for the upper / lower / surface pieces, each with the same solver settings as `_evt`."""
    iv = PB._prep_inversion(*e.ev, *e.cl, e.band, GRID)
    q_m = np.asarray(iv.q_m, np.float64)
    bad = np.abs(q_m - 9999.90) < 1e-2
    thb = np.asarray(iv.THBIN, np.float64)
    dth_tot = np.asarray(iv.THTIN, np.float64) - thb

    qwu_a = (wu_pv_interior(*e.ev, lat_n=e.band[0], dlat=GRID, dlon=GRID)
             - wu_pv_interior(*e.cl, lat_n=e.band[0], dlat=GRID, dlon=GRID))

    def core3(a9):
        o = np.transpose(np.asarray(a9, np.float64), (1, 2, 0))
        return np.where(bad, 0.0, np.where(np.isfinite(o), o, 0.0))

    pd = PassDParams(thrsh=thrsh)

    def one(dq, dth, qlv_list):
        qlv = np.zeros(NL, np.int32)
        qlv[:len(qlv_list)] = qlv_list
        _, SP = iv.ext.qinvertp_core(
            iv.MBIN, iv.SBIN, iv.H_bal, iv.SI_bal, iv.QBIN,
            np.asfortranarray(q_m + dq, np.float32), iv.THBIN,
            np.asfortranarray(thb + dth, np.float32),
            iv.zhdr8, PB.PR, qlv, len(qlv_list),
            np.float32(pd.omegs), np.float32(pd.omegah), np.float32(pd.part),
            np.float32(pd.thrsh), np.float32(pd.tscal), np.float32(pd.qscal),
            int(pd.inlin), int(pd.ibc))
        return _from_core(SP) * 1e5

    out, zero = {}, np.zeros_like(dth_tot)
    # upper: interior PV on 400-200 plus the TOP theta at 100 hPa
    dq_u = np.zeros_like(qwu_a); dq_u[_evt.UPPER_IDX[:-1]] = qwu_a[_evt.UPPER_IDX[:-1]]
    dth_u = np.zeros_like(dth_tot); dth_u[:, :, 1] = dth_tot[:, :, 1]
    # lower: interior PV on 850-500, no theta at all
    dq_l = np.zeros_like(qwu_a); dq_l[[_evt.WU.index(p) for p in (850, 700, 500)]] = \
        qwu_a[[_evt.WU.index(p) for p in (850, 700, 500)]]
    # surface: the BOTTOM theta only
    dth_s = np.zeros_like(dth_tot); dth_s[:, :, 0] = dth_tot[:, :, 0]

    for name, dq, dth, qlv in (("upper", core3(dq_u), dth_u, [5, 6, 7, 8, 9]),
                               ("lower", core3(dq_l), zero, [2, 3, 4]),
                               ("surface", core3(np.zeros_like(qwu_a)), dth_s, [1])):
        t = time.time()
        out[name] = one(dq, dth, qlv)
        print(f"  {name:8s} inverted in {time.time()-t:6.1f} s")
    return out


def main():
    rec = _evt.pick_event(seed=0)
    e = _evt.load(rec)
    z = np.load(HERE / "_cache.npz", allow_pickle=True)

    psi = invert_pieces(e)
    W = {k: _evt.winds(e, v) for k, v in psi.items()}
    kw = _evt.WIND_HPA.index(250)
    kr = 1                                              # 250 hPa within the wavg triple

    ur, vr = z["u_rot_anom"][kr], z["v_rot_anom"][kr]    # EXTERNAL reference
    uu, vu = W["upper"][0][kw], W["upper"][1][kw]
    ul, vl = W["lower"][0][kw], W["lower"][1][kw]
    us, vs = W["surface"][0][kw], W["surface"][1][kw]

    resid = (uu - ur, vu - vr)                          # = 03 panel (d)
    lowsurf = (ul + us, vl + vs)                         # what should explain -(d)

    rms = lambda a, b: float(np.sqrt(np.nanmean(a ** 2 + b ** 2)))

    def corr(a, b, c, d):
        x = np.concatenate([a.ravel(), b.ravel()])
        y = np.concatenate([c.ravel(), d.ravel()])
        m = np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[m], y[m])[0, 1])

    print()
    print("  250 hPa RMS |V'|  [m/s]")
    print(f"    V_rot      (EXTERNAL reference)        {rms(ur, vr):7.3f}")
    print(f"    V_upper    (what 03 inverts)           {rms(uu, vu):7.3f}")
    print(f"    V_lower                                {rms(ul, vl):7.3f}")
    print(f"    V_surface                              {rms(us, vs):7.3f}")
    print(f"    03 panel (d) = V_upper - V_rot         {rms(*resid):7.3f}"
          f"   ({rms(*resid)/rms(ur, vr):.0%} of the reference)")
    print()
    fullsum = (uu + ul + us, vu + vl + vs)
    unexpl = (fullsum[0] - ur, fullsum[1] - vr)
    print(f"    -(d) should be V_lower + V_surface:")
    print(f"      RMS(V_lower+V_surface)               {rms(*lowsurf):7.3f}")
    print(f"      corr with -(d)                       {corr(-resid[0], -resid[1], *lowsurf):7.4f}")
    print()
    print(f"    ALL pieces summed vs the reference:")
    print(f"      RMS(V_upper+V_lower+V_surface)       {rms(*fullsum):7.3f}")
    print(f"      RMS(sum - V_rot)  = the FLOOR        {rms(*unexpl):7.3f}"
          f"   ({rms(*unexpl)/rms(ur, vr):.0%} of the reference)")
    print(f"      corr(sum, V_rot)                     {corr(*fullsum, ur, vr):7.4f}")
    print()
    print("  The floor is what nonlinear balance cannot represent (unbalanced rotational flow)")
    print("  plus solver error. It does not shrink with more iterations or more pieces.")


if __name__ == "__main__":
    main()
