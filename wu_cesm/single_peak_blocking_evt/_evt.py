#!/usr/bin/env python3
"""Shared machinery for figures 01-04: pick the event, build the object, run the PPVI.

Every script in this folder imports from here so they cannot drift apart on the event, the
threshold, the level sets, or the inversion settings.

THE SPEC (final, 2026-08-06)
----------------------------
pieces      lower = K{2,3,4} = 850/700/500 ; upper = K{5..9} = 400/300/250/200 + 100 (theta)
object      3-D, 6-connectivity, on 400/300/250/200 (100 hPa EXCLUDED — the archive PV there is
            ~47% missing and the gaps are worst at high latitude), found on the k<=4 field,
            threshold f = 0.25 of the seed value, NO smoothing (it created object where the binary
            object had none)
q'_p        M * Pi_{k<=4}[q']      (order (b): filter first, then find the object)
q'_e        q' - q'_p
100 hPa     NOT part of the object; it still contributes the top boundary theta to the inversion,
            masked by M at 200 hPa
PPVI        upper piece as the ONLY source; winds written at 400/300/250/200

TWO PV VARIABLES, DELIBERATELY
------------------------------
The object and the stored `pv_anom_p/e` live in the ARCHIVE Ertel PV, because that is what the
budget differentiates in grad q'. The inversion is fed the WU CORE's own PV, because the two are
not pointwise interchangeable (slope 79-94 instead of the exact 100, r = 0.83-0.99 — the core
re-differentiates theta over only 9 coarse pressure levels). The SAME mask M is applied to both.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy import ndimage

sys.path.insert(0, "/net/flood/data2/users/x_yan/cesm-blocking/08_triad_resonance")
sys.path.insert(0, "/net/flood/data2/users/x_yan/pvtend/paper/grl_paper_part1/"
                   "fig4_rot_nl_block_vs_prp_talia_style")
import build_pvbudget_15deg as Bg                                    # noqa: E402
import ppvi_block_wind as PB                                        # noqa: E402
from wave_diagnostic import zonal_bandpass                          # noqa: E402
from pvtend.ppvi.solver import (fill_below_ground, PassDParams, _from_core, NL)  # noqa: E402
from pvtend.ppvi.winds import psi_to_winds                          # noqa: E402
from wu_pv import wu_pv_interior                                    # noqa: E402

WU = list(Bg.WU)                                     # [1000,850,700,500,400,300,250,200,100]
UPPER_HPA = [400, 300, 250, 200, 100]                # Wu piece `upper`, K=5..9 (PPVI source set)
UPPER_IDX = [WU.index(p) for p in UPPER_HPA]
#: The OBJECT is built on 400/300/250/200 only — 100 hPa is EXCLUDED.
#: `pv9_15deg` has only ~47% valid PV at 100 hPa and the gaps are latitude-dependent, worst exactly
#: where blocks live (0% at 85.5N, ~50% at 55N). Seeding a mask on a half-missing field would make
#: the object's top depend on where the archive happens to have data. 100 hPa still enters the
#: inversion as the top boundary theta, masked by M at 200 hPa.
OBJ_HPA = [400, 300, 250, 200]
OBJ_IDX = [WU.index(p) for p in OBJ_HPA]
SEED_LEV_IN_OBJ = OBJ_HPA.index(250)
TOPTH_SRC = OBJ_HPA.index(200)                       # which object level masks the 100 hPa top theta
WIND_HPA = [400, 300, 250, 200]                      # interior only: 100 hPa carries no interior PV
WIND_IDX = [WU.index(p) for p in WIND_HPA]
UPPER_QLV = [5, 6, 7, 8, 9]                          # 1-based, what BALP activates as the source
F_THRESH = 0.25
GRID = 1.5
#: Wu pseudo-PV -> PVU. Algebraically exact (Q_wu = 1e8 * PV_SI = 100 * PV[PVU]); the MEASURED
#: slope is 79-94 because the core re-differentiates theta over only 9 coarse pressure levels.
WU2PVU = 1.0e-2
STRUCT6 = ndimage.generate_binary_structure(3, 1)    # 6-connectivity in (level, y, x)


@dataclass
class Event:
    meta: dict
    band: np.ndarray            # band latitudes, N->S
    iw: np.ndarray              # longitude window indices into the global ring
    jcen: int
    icen: int
    ev: list                    # [z,t,u,v] window cubes, 9 levels, below-ground filled
    cl: list
    pv_anom: np.ndarray         # ARCHIVE PV anomaly, 9 levels, window  [PVU]
    k4: np.ndarray              # Pi_{k<=4}[pv_anom], 9 levels, window  [PVU]
    M: np.ndarray               # binary object weight on OBJ_HPA = 400/300/250/200, (4, NY, NX)
    obj: np.ndarray             # the binary object before smoothing
    info: dict


def _k4_global_then_crop(a9_global, iw):
    """Pi_{k in [0,4]} on the FULL ring, then crop. NaN-safe (an rFFT over a NaN row is all-NaN)."""
    bad = ~np.isfinite(a9_global)
    out = zonal_bandpass(np.where(bad, 0.0, a9_global), m_min=0, m_max=4, lon_axis=-1)
    return np.where(bad, np.nan, out)[..., iw]


def pick_event(seed=0, group="block", stage="peak"):
    """A reproducible random event. dh=0 is the only slice the rebuild covers."""
    allrows, _ = Bg.build_tasks([group])
    d = allrows[(allrows["group"] == group) & (allrows["stage"] == stage)]
    rng = np.random.default_rng(seed)
    return d.iloc[int(rng.integers(len(d)))].to_dict()


def load(rec):
    Bg._load_clim15()
    mem, dec = int(rec["member"]), str(rec["decade"])
    fp = Bg.PVDIR / f"m{mem}" / f"cesm2_lens2_pv9_15deg_m{mem:02d}_d{dec}.nc"
    with xr.open_dataset(fp) as ds:
        lat, lon = ds["lat"].values, ds["lon"].values
        didx = {(int(t.year), int(t.month), int(t.day)): i for i, t in enumerate(ds["time"].values)}
        ti = didx[(int(rec["year"]), int(rec["month"]), int(rec["day"]))]
        pv9 = ds["pv"].sel(lev=WU).isel(time=ti).load().values
        ev9 = [ds[v].sel(lev=WU).isel(time=ti).load().values for v in ("z", "t", "u", "v")]
    jb = np.where((lat >= Bg.BAND_S) & (lat <= Bg.BAND_N))[0][::-1]
    band, nlon = lat[jb], lon.size
    mo, di = int(rec["month"]), int(rec["day"]) - 1
    cm = Bg._CLIM15[mo]
    clat, clon = float(rec["lat"]), float(rec["lon180"]) % 360.0
    ic = int(np.abs(((lon - clon + 180) % 360) - 180).argmin())
    iw = (np.arange(-Bg.WIN_PAD, Bg.WIN_PAD + 1) + ic) % nlon
    jcen = int(np.abs(band - clat).argmin())

    anom_g = (pv9 - cm["pv"][di])[:, jb, :].astype(np.float64)       # GLOBAL longitudes
    pv_anom = anom_g[..., iw]
    k4 = _k4_global_then_crop(anom_g, iw)
    ev = list(fill_below_ground(*[c[:, jb, :].astype(np.float64)[..., iw] for c in ev9]))
    cl = list(fill_below_ground(*[cm[v][di][:, jb, :].astype(np.float64)[..., iw]
                                  for v in ("z", "t", "u", "v")]))
    M, obj, info = build_object(k4[OBJ_IDX], jcen, Bg.WIN_PAD)
    meta = dict(member=mem, decade=dec, year=int(rec["year"]), month=mo, day=di + 1,
                stage=str(rec["stage"]), label=str(rec["label"]), track=int(rec["track"]),
                clat=clat, clon=clon)
    return Event(meta, band, iw, jcen, Bg.WIN_PAD, ev, cl, pv_anom, k4, M, obj, info)


def build_object(q4_obj, jcen, icen, f=F_THRESH, search_rad=8):
    """6-connected 3-D negative object containing the seed, on OBJ_HPA of the k<=4 field.

    NO smoothing. A Gaussian filter was used originally; it was removed because it CREATED object
    where the binary object had none — on the test event it produced 70 points with M>0.5 at 400 hPa
    while the binary object there was empty, by bleeding the 300 hPa object downwards. The mask is
    now the binary object itself, as a float.
    """
    seed = q4_obj[SEED_LEV_IN_OBJ, jcen, icen]
    if not np.isfinite(seed) or seed >= 0:
        j0, j1 = max(jcen - search_rad, 0), min(jcen + search_rad + 1, q4_obj.shape[1])
        i0, i1 = max(icen - search_rad, 0), min(icen + search_rad + 1, q4_obj.shape[2])
        sub = q4_obj[SEED_LEV_IN_OBJ, j0:j1, i0:i1]
        dj, di = np.unravel_index(np.nanargmin(np.where(np.isfinite(sub), sub, np.inf)), sub.shape)
        jcen, icen = j0 + dj, i0 + di
        seed = q4_obj[SEED_LEV_IN_OBJ, jcen, icen]
    binary = np.isfinite(q4_obj) & (q4_obj < f * seed)
    lab, _ = ndimage.label(binary, structure=STRUCT6)
    lb = lab[SEED_LEV_IN_OBJ, jcen, icen]
    obj = (lab == lb) if lb > 0 else np.zeros_like(binary)
    M = obj.astype(float)                       # NO smoothing — see the docstring
    levs = [OBJ_HPA[k] for k in range(obj.shape[0]) if obj[k].any()]
    return M, obj, dict(seed_val=float(seed), seed_ji=(jcen, icen), levels=levs,
                        nlev=len(levs), area_frac=float(obj.mean()), f=f)


def split_pv(e: Event):
    """(pv_anom_p, pv_anom_e) in the ARCHIVE PV, 9 levels. `_p` is nonzero on the UPPER levels only."""
    p = np.zeros_like(e.pv_anom)
    p[OBJ_IDX] = e.M * e.k4[OBJ_IDX]
    return p, e.pv_anom - p


def invert(e: Event, thrsh=1e-1):
    """PPVI with the UPPER piece as the ONLY source. Returns psi for 'p', 'e' and 'tot' (9,NY,NX).

    `thrsh=1e-1` is `PassDParams`' default and the production setting. Measured on this event, the
    whole range 1e-1 .. 1e-3 gives the same solution (wind within 0.8%, psi within 0.25%, external
    corr(V_tot, V_rot) identical to three decimals) for 4.6x the cost, so there is nothing to buy by
    tightening. `1e-4` is BROKEN outright — larger psi, exactly zero wind, i.e. psi collapsed to a
    constant. The figures in this folder were rendered at 1e-3 before this was measured; they are
    unaffected at plotting resolution.

    Fed in the Wu core's own PV units, masked by the SAME M that defined the archive-PV object.
    100 hPa contributes only boundary theta, exactly as the Wu algorithm has it.
    """
    iv = PB._prep_inversion(*e.ev, *e.cl, e.band, GRID)
    q_m = np.asarray(iv.q_m, np.float64)
    bad = np.abs(q_m - 9999.90) < 1e-2
    thb = np.asarray(iv.THBIN, np.float64)
    dth_tot = np.asarray(iv.THTIN, np.float64) - thb

    lat_n = e.band[0]
    qwu_a = (wu_pv_interior(*e.ev, lat_n=lat_n, dlat=GRID, dlon=GRID)
             - wu_pv_interior(*e.cl, lat_n=lat_n, dlat=GRID, dlon=GRID))
    qwu_k4 = np.where(np.isfinite(qwu_a),
                      zonal_bandpass(np.where(np.isfinite(qwu_a), qwu_a, 0.0), 0, 4, -1), np.nan)

    def core3(a9):
        o = np.transpose(np.asarray(a9, np.float64), (1, 2, 0))
        return np.where(bad, 0.0, np.where(np.isfinite(o), o, 0.0))

    dq_tot = core3(qwu_a)
    dq_p9 = np.zeros_like(qwu_a)
    dq_p9[OBJ_IDX] = e.M * qwu_k4[OBJ_IDX]            # M applied to the WU PV, same object
    dq_p = core3(dq_p9)
    # top theta (100 hPa) masked by M at ITS OWN level, per the spec; bottom theta is not in `upper`
    dth_p = np.zeros_like(dth_tot)
    dth_p[:, :, 1] = e.M[TOPTH_SRC] * dth_tot[:, :, 1]   # 100 hPa top theta, masked at 200 hPa

    pd = PassDParams(thrsh=thrsh)
    qlv = np.zeros(NL, np.int32)
    qlv[:len(UPPER_QLV)] = UPPER_QLV

    def one(dq, dth):
        _, SP = iv.ext.qinvertp_core(
            iv.MBIN, iv.SBIN, iv.H_bal, iv.SI_bal, iv.QBIN,
            np.asfortranarray(q_m + dq, np.float32), iv.THBIN,
            np.asfortranarray(thb + dth, np.float32),
            iv.zhdr8, PB.PR, qlv, len(UPPER_QLV),
            np.float32(pd.omegs), np.float32(pd.omegah), np.float32(pd.part),
            np.float32(pd.thrsh), np.float32(pd.tscal), np.float32(pd.qscal),
            int(pd.inlin), int(pd.ibc))
        return _from_core(SP) * 1e5

    psi_p = one(dq_p, dth_p)
    psi_tot = one(dq_tot, dth_tot)
    return {"p": psi_p, "e": psi_tot - psi_p, "tot": psi_tot}


def wu_pv_split(e: Event):
    """The Wu core's own PV anomaly, split by the SAME mask M, in PVU. (tot, p, e), 9 levels.

    Cheap — two `wu_pv_interior` calls and a bandpass, no inversion — so 04 can call it without
    forcing a cache rebuild. This is the field the INVERSION was fed; `split_pv` returns the field
    the BUDGET differentiates. They are not the same variable and 04 exists to quantify that.

    Levels 1 and 9 (1000/100 hPa) carry no interior PV in the Wu formulation and are returned as
    whatever the routine puts there — compare only on 850-200.
    """
    lat_n = e.band[0]
    qa = (wu_pv_interior(*e.ev, lat_n=lat_n, dlat=GRID, dlon=GRID)
          - wu_pv_interior(*e.cl, lat_n=lat_n, dlat=GRID, dlon=GRID))
    qk4 = np.where(np.isfinite(qa), zonal_bandpass(np.where(np.isfinite(qa), qa, 0.0), 0, 4, -1),
                   np.nan)
    qp = np.zeros_like(qa)
    qp[OBJ_IDX] = e.M * qk4[OBJ_IDX]
    return qa * WU2PVU, qp * WU2PVU, (qa - qp) * WU2PVU


def winds(e: Event, psi):
    """(u, v) at 400/300/250/200 from a 9-level psi."""
    u, v = psi_to_winds(psi, e.band, GRID, GRID)
    return u[WIND_IDX], v[WIND_IDX]


def side_colorbar(fig, mappable, axes, label, pad=0.018, width=0.015, shrink=1.0, **kw):
    """Colourbar in an explicitly positioned axes to the right of `axes`. Call AFTER subplots_adjust.

    `fig.colorbar(..., ax=axes)` steals space from the axes it is handed, and whether the result
    overlaps depends on the order of that call relative to `subplots_adjust` and on `fraction`/`pad`
    guesses that do not know the panel aspect. Explicit placement removes the guessing: the bar is
    put at `pad` to the right of the rightmost panel edge and spans the panels' vertical extent.
    Its extend triangles then scale with `width`, so they stay small instead of ballooning.
    """
    import numpy as _np
    bb = [ax.get_position() for ax in _np.atleast_1d(_np.asarray(axes, dtype=object)).ravel()]
    x1 = max(b.x1 for b in bb)
    y0, y1 = min(b.y0 for b in bb), max(b.y1 for b in bb)
    h = (y1 - y0) * shrink
    cax = fig.add_axes([x1 + pad, y0 + (y1 - y0 - h) / 2, width, h])
    cb = fig.colorbar(mappable, cax=cax, **kw)
    cb.set_label(label)
    cb.ax.tick_params(labelsize=8.5)
    return cb


def rot_anom_reference(e: Event, rec):
    """Helmholtz rotational-wind anomaly from the archive — an EXTERNAL check on the total."""
    mem, dec = int(rec["member"]), str(rec["decade"])
    fp = Bg.PVDIR / f"m{mem}" / f"cesm2_lens2_pv9_15deg_m{mem:02d}_d{dec}_rot.nc"
    with xr.open_dataset(fp) as dr:
        didx = {(int(t.year), int(t.month), int(t.day)): i for i, t in enumerate(dr["time"].values)}
        ti = didx[(int(rec["year"]), int(rec["month"]), int(rec["day"]))]
        ur = dr["u_rot"].isel(time=ti).load().values
        vr = dr["v_rot"].isel(time=ti).load().values
    lat = np.linspace(-90, 90, 121)
    jb = np.where((lat >= Bg.BAND_S) & (lat <= Bg.BAND_N))[0][::-1]
    cm = Bg._CLIM15[e.meta["month"]]
    di = e.meta["day"] - 1
    ua = (ur - cm["u_rot_bar"][di][list(Bg.WAVG_IDX)])[:, jb, :][..., e.iw]
    va = (vr - cm["v_rot_bar"][di][list(Bg.WAVG_IDX)])[:, jb, :][..., e.iw]
    return ua, va                                    # (3, NY, NX) at the wavg levels 300/250/200
