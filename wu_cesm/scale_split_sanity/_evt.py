"""Shared event loader for the scale-split sanity checks.

Loads one CESM2-LENS2 smbb event (state cached by
`wu_cesm/test_one_case_derecho.py`), selects the 9 Wu levels, and runs pass A/B
on the **full 360-degree** NH band to get Wu PV and boundary theta globally --
which is what the zonal wavenumber filter needs.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/glade/derecho/scratch/kenyan/01_cesm_processing/05_pv_budget_closure")

import lens2_archive as la          # noqa: E402
from scale_split import wu_state_band   # noqa: E402

CACHE_DIR = "/glade/work/kenyan/pv_inversion_isentropic/wu_cesm"
WU9_HPA = np.array([1000., 850., 700., 500., 400., 300., 250., 200., 100.])
BAND_S, BAND_N = 10.5, 85.5

# 0-based indices into the 9-level Wu grid
K_SURFACE = 0                      # 1000 hPa bottom boundary theta
K_LOWER = [1, 2, 3]                # 850, 700, 500
K_UPPER_INT = [4, 5, 6, 7]         # 400, 300, 250, 200  (interior PV)
K_TOP = 8                          # 100 hPa top boundary theta


def load_event(cache=None, member=91):
    """Return a dict with the global NH-band Wu state of one event."""
    cache = cache or sorted(glob.glob(f"{CACHE_DIR}/_state_cache_*.npz"))[0]
    z = np.load(cache)

    # the cache was written on the 20-level grid; select the 9 Wu levels
    plev20 = np.array([1000., 950., 900., 850., 800., 750., 700., 650., 600., 550.,
                       500., 450., 400., 350., 300., 250., 200., 150., 100., 50.])
    sel = [int(np.argmin(np.abs(plev20 - p))) for p in WU9_HPA]
    ev = {v: z[f"ev_{v}"][sel] for v in ("Z3", "T", "U", "V")}
    mn = {v: z[f"mn_{v}"][sel] for v in ("Z3", "T", "U", "V")}

    lat, lon, *_ = la.grid_and_hybrid("smbb", member)
    band = np.nonzero((lat >= BAND_S - 1e-6) & (lat <= BAND_N + 1e-6))[0][::-1]  # N->S
    band_lats = lat[band]
    dlat = float(abs(lat[1] - lat[0]))
    dlon = float(lon[1] - lon[0])
    nx = len(lon)

    cut = lambda a: np.ascontiguousarray(a[:, band, :])
    He, Te, Ue, Ve = (cut(ev[v]) for v in ("Z3", "T", "U", "V"))
    Hm, Tm, Um, Vm = (cut(mn[v]) for v in ("Z3", "T", "U", "V"))

    q_e, thb_e, tht_e = wu_state_band(He, Te, Ue, Ve,
                                      band_lats[0], band_lats[-1], dlat, dlon, nx)
    q_m, thb_m, tht_m = wu_state_band(Hm, Tm, Um, Vm,
                                      band_lats[0], band_lats[-1], dlat, dlon, nx)

    return dict(cache=cache, lat=lat, lon=lon, band=band, band_lats=band_lats,
                dlat=dlat, dlon=dlon, nx=nx,
                He=He, Te=Te, Ue=Ue, Ve=Ve, Hm=Hm, Tm=Tm, Um=Um, Vm=Vm,
                q_e=q_e, q_m=q_m, q_anom=q_e - q_m,
                th_top_e=tht_e, th_top_m=tht_m, th_top_anom=tht_e - tht_m,
                th_bot_anom=thb_e - thb_m)


def event_label(cache):
    tag = os.path.basename(cache).replace("_state_cache_", "").replace(".npz", "")
    m, dt = tag.split("_")
    return f"member {m[1:]}  {dt[:4]}-{dt[4:6]}-{dt[6:8]} {dt[8:10]}Z"
