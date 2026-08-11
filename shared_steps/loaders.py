"""Per-dataset state loaders — everything reduced to the same 9-level Wu cube.

Each loader returns ``(H, T, U, V)`` on the 9 Wu pressure levels, on the NH band
10.5-85.5 N, **global in longitude** (the zonal wavenumber filter needs the full
circle), N->S in latitude, plus the grid metadata.

The four sources differ only here:

    cesm6h   CESM2-LENS2 smbb members 91-100, hour_6 cam.h5, 32 hybrid levels
             p = hyam*P0 + hybm*ps          (hyam dimensionless, top -> surface)
    cesmday  CESM2-LENS2 cmip6 members 1-10, day_1 cam.h6, daily MEAN
             same hybrid convention
    jra3q    JRA-3Q anl_mdl, 100 hybrid levels, 480x960 GAUSSIAN
             p = a + b*ps                   (a in Pa, surface -> top)  <-- different!
    era5     RDA d633000 e5.oper.an.pl, already isobaric, 0.25 deg
             no hybrid conversion at all

The JRA convention is deliberately different from CAM: getting `a` wrong fails by
~1e9 Pa and is loud, but getting the LEVEL ORDER wrong fails silently, because
the interpolators only require monotonicity, not a direction.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
sys.path.insert(0, "/glade/derecho/scratch/kenyan/01_cesm_processing/06_isentropic_clim")
sys.path.insert(0, "/glade/derecho/scratch/kenyan/01_cesm_processing/05_pv_budget_closure")

WU9_HPA = np.array([1000., 850., 700., 500., 400., 300., 250., 200., 100.])
WU9_PA = WU9_HPA * 100.0
BAND_S, BAND_N = 10.5, 85.5

# Target grid spacing [deg] for the inversion.  ERA5 (0.25) and JRA-3Q (0.375) are
# strided down to ~1 deg; CESM f09 (0.94 x 1.25) is left alone.  Two reasons: the
# Wu box is an SOR solve, and at 0.25 deg a +-90 deg box is 721 x 301 points --
# 28x the CESM box and past the develop walltime; and comparing four datasets is
# only meaningful at a common resolution.
TARGET_DEG = 1.0

JRA = "/glade/campaign/collections/rda/data/d640000"
ERA5 = "/glade/campaign/collections/rda/data/d633000"


def _band(lat):
    """Indices of the 10.5-85.5 N band, ordered N->S."""
    return np.nonzero((lat >= BAND_S - 1e-6) & (lat <= BAND_N + 1e-6))[0][::-1]


def _pack(H, T, U, V, lat, lon):
    """Cut to the NH band and, if the latitudes are not equally spaced, resample
    them onto a uniform grid.

    The Wu solver's header carries a single `dlat`, so it assumes a regular grid.
    JRA-3Q's model levels only exist on a 480-point GAUSSIAN grid whose spacing
    varies by ~1 % (0.3715 deg at the band edge vs ~0.375 in mid-latitudes), which
    would be silently mis-differenced.  One linear resample of the band onto
    uniform latitudes fixes it, and makes the spherical-harmonic path `regular`
    for every dataset.
    """
    slat = max(1, int(round(TARGET_DEG / abs(lat[1] - lat[0]))))
    slon = max(1, int(round(TARGET_DEG / abs(lon[1] - lon[0]))))
    if slat > 1 or slon > 1:
        lat = lat[::slat]
        lon = lon[::slon]
        H, T, U, V = (np.asarray(a)[:, ::slat, ::slon] for a in (H, T, U, V))

    U_g, V_g = np.asarray(U), np.asarray(V)          # keep the global wind
    b = _band(lat)
    blat = lat[b]
    cut = lambda a: np.ascontiguousarray(np.asarray(a, dtype=np.float64)[:, b, :])
    H, T, U, V = cut(H), cut(T), cut(U), cut(V)

    d = np.diff(blat)
    if not np.allclose(d, d[0], rtol=2e-3, atol=1e-6):
        new = np.linspace(blat[0], blat[-1], len(blat))
        order = np.argsort(blat)
        def rs(a):
            out = np.empty_like(a)
            for k in range(a.shape[0]):
                for i in range(a.shape[2]):
                    out[k, :, i] = np.interp(new, blat[order], a[k, order, i])
            return out
        H, T, U, V = rs(H), rs(T), rs(U), rs(V)
        blat = new

    return dict(H=H, T=T, U=U, V=V,
                # the GLOBAL wind is kept as well: the Helmholtz decomposition has
                # to run on a real sphere.  `sh_ops.is_nh_grid` only recognises a
                # hemisphere that runs from the equator to the pole, so the
                # 10.5-85.5 N band is silently treated as if its 80 rows spanned
                # the full 180 deg -- every meridional derivative then comes out
                # scaled by ~180/74.  A band is not a sphere and cannot be made
                # into one, so the transform is done globally and cropped after.
                U_glob=np.asarray(U_g, dtype=np.float64),
                V_glob=np.asarray(V_g, dtype=np.float64),
                lat=lat, lon=lon, band=b, band_lats=blat,
                dlat=float(abs(blat[1] - blat[0])), dlon=float(lon[1] - lon[0]),
                nx=len(lon))


# ── CESM (both cadences share the hybrid path) ─────────────────────────────
def _cesm(member, when, set_key):
    import cftime
    import lens2_archive as la
    import state_isobaric as si

    ms = la.MEMBER_SETS[set_key]
    lat, lon, hyam, hybm, p0 = la.grid_and_hybrid(set_key, member)
    ti = la.time_index(set_key, member, "U")
    k = ti.index_of(cftime.DatetimeNoLeap(when.year, when.month, when.day,
                                          getattr(when, "hour", 0)))
    F = {v: la.read_slice(ms, member, v, k) for v in ("U", "V", "T", "Q", "Z3", "OMEGA")}
    ps = la.read_slice(ms, member, "PS", k)
    st = si.state_on_pressure(F, ps, hyam, hybm, p0, lat, lon)
    return _pack(st["Z3"], st["T"], st["U"], st["V"], lat, lon)


def load_cesm6h(when, member=91):
    return _cesm(member, when, "smbb")


def load_cesmday(when, member=1):
    return _cesm(member, when, "cmip6")


# ── JRA-3Q: p = a + b*ps, a in Pa, levels surface -> top, Gaussian grid ─────
def _jra_open(pattern, when):
    import cftime
    import netCDF4 as nc
    for f in sorted(glob.glob(pattern)):
        d = nc.Dataset(f)
        tv = d.variables["time"]
        dates = cftime.num2date(tv[:], tv.units, getattr(tv, "calendar", "standard"))
        hit = np.nonzero(np.array([str(x)[:13] for x in dates]) == str(when)[:13])[0]
        if hit.size:
            return d, int(hit[0])
        d.close()
    raise SystemExit(f"{when} not found in {pattern}")


def _m2n(x):
    """netCDF4 masked array -> float64 with NaN, never the raw ~1e20 fill value."""
    return np.ma.filled(np.ma.masked_invalid(np.ma.asarray(x)).astype(np.float64), np.nan)


def load_jra3q(when, member=None):
    import netCDF4 as nc
    sys.path.insert(0, "/glade/derecho/scratch/kenyan/01_cesm_processing/06_isentropic_clim")
    from pv_isentropic import interp_monotonic

    ym = f"{when.year:04d}{when.month:02d}"
    got = {}
    for var, tag in (("tmp", "tmp"), ("ugrd", "ugrd"), ("vgrd", "vgrd"), ("hgt", "hgt")):
        d, k = _jra_open(f"{JRA}/anl_mdl/{ym}/*{tag}-hyb-an-gauss*", when)
        key = [v for v in d.variables if v.endswith("-hyb-an-gauss")][0]
        got[var] = _m2n(d.variables[key][k])
        if var == "tmp":
            a = np.asarray(d.variables["a_hybrid_level"][:], dtype=np.float64)   # Pa
            b = np.asarray(d.variables["b_hybrid_level"][:], dtype=np.float64)
            lat = np.asarray(d.variables["lat"][:], dtype=np.float64)
            lon = np.asarray(d.variables["lon"][:], dtype=np.float64)
        d.close()
    dp, kp = _jra_open(f"{JRA}/anl_surf/{ym}/*pres-sfc-an-gauss*", when)
    ps = _m2n(dp.variables[[v for v in dp.variables if v.startswith("pres")][0]][kp])
    dp.close()

    if lat[0] > lat[-1]:                       # JRA ships N->S; work S->N like CAM
        lat = lat[::-1]
        for v in got:
            got[v] = got[v][:, ::-1, :]
        ps = ps[::-1, :]

    p3d = a[:, None, None] + b[:, None, None] * ps[None, :, :]
    out = {k: interp_monotonic(v, p3d, WU9_PA, log=True) for k, v in got.items()}
    return _pack(out["hgt"], out["tmp"], out["ugrd"], out["vgrd"], lat, lon)


# ── ERA5: already isobaric, just select the 9 levels ───────────────────────
def load_era5(when, member=None):
    import netCDF4 as nc
    ymd = f"{when.year:04d}{when.month:02d}"
    got, lat, lon = {}, None, None
    NAME = {"z": "Z", "t": "T", "u": "U", "v": "V"}
    for short, code in (("z", "128_129_z"), ("t", "128_130_t"),
                        ("u", "128_131_u"), ("v", "128_132_v")):
        pat = f"{ERA5}/e5.oper.an.pl/{ymd}/*{code}*{when.year:04d}{when.month:02d}{when.day:02d}*.nc"
        f = sorted(glob.glob(pat))
        if not f:
            raise SystemExit(f"no ERA5 file for {short}: {pat}")
        d = nc.Dataset(f[0])
        lev = np.asarray(d.variables["level"][:], dtype=float)
        sel = [int(np.argmin(np.abs(lev - p))) for p in WU9_HPA]
        tv = d.variables["time"]
        import cftime
        dates = cftime.num2date(tv[:], tv.units, getattr(tv, "calendar", "standard"))
        it = int(np.argmin([abs((x - when).total_seconds()) for x in dates]))
        key = [v for v in d.variables if v.upper() == NAME[short]][0]
        got[short] = _m2n(d.variables[key][it, sel, :, :])
        lat = np.asarray(d.variables["latitude"][:], dtype=float)
        lon = np.asarray(d.variables["longitude"][:], dtype=float)
        d.close()
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        for v in got:
            got[v] = got[v][:, ::-1, :]
    got["z"] = got["z"] / 9.80665          # ERA5 ships geopotential, we need height
    return _pack(got["z"], got["t"], got["u"], got["v"], lat, lon)


LOADERS = {"cesm6h": load_cesm6h, "cesmday": load_cesmday,
           "jra3q": load_jra3q, "era5": load_era5}
