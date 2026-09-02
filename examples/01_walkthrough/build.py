"""Assemble the walkthrough notebook.

The notebook is generated rather than written by hand so that its figures, the
numbers quoted in the documentation, and the code that produced them cannot drift
apart. Run this, then execute the notebook:

    micromamba run -n blocking python examples/01_walkthrough/build.py
    micromamba run -n blocking jupyter nbconvert --to notebook --execute \
        --inplace --ExecutePreprocessor.timeout=3600 \
        examples/01_walkthrough/01_walkthrough.ipynb

Figures land in ``figures/`` and are the ones the documentation includes.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ---------------------------------------------------------------------------

md(r"""
# Piecewise potential vorticity inversion on the sphere

This notebook runs one inversion from beginning to end and plots what each step
produced. It uses the ERA5 blocking case that ships with the package.

The question the inversion answers is: **of the anomalous rotational wind at
250 hPa, how much was induced by the potential vorticity anomaly at each level,
and how much by the potential temperature anomaly at the ground?**

The steps are

1. the event and the climatology it is an anomaly against
2. the vertical coordinate, and potential temperature
3. the streamfunction, from the wind
4. Ertel potential vorticity
5. the anomaly that will be inverted
6. extending the hemisphere to the sphere
7. the solve
8. the wind induced by each level
9. the same, grouped into surface, lower and upper
10. what the pieces add up to, and what is left over

Every figure is written to `figures/` and is the one the documentation shows.
""")

code(r"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path.cwd().parents[1] / "src"))

from pvinv_sph.config import InversionConfig, MirrorConfig
from pvinv_sph.levels import CP, KAPPA, P0, build_levels
from pvinv_sph.passab import ertel_pv_pvu
from pvinv_sph.passd import invert_pieces
from pvinv_sph.prepare import prepare_state
from pvinv_sph.sht import SHT, gaussian_grid
from pvinv_sph.sphere import SphereOps
from pvinv_sph.winds import rotational_wind_stack

DATA = Path.cwd().parents[1] / "data"
FIGURES = Path.cwd() / "figures"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "savefig.bbox": "tight"})
TARGET = 250.0            # the level the induced wind is reported on
CASE = "era5_blocking"
DISC = 40.0               # radius of the disc drawn about an event, in degrees
""")

code(r"""
# Every map in this notebook is drawn about the event rather than about the
# earth's axis: the frame is rotated so the event is at the pole of the
# projection, and a disc of fixed great-circle radius is shown.  Every event then
# has the same geometry whatever its latitude, and the earth's pole is an ordinary
# point inside the picture rather than a place where the coordinates fail.
import cartopy.crs as ccrs
import matplotlib.path as mpath

_CIRCLE = mpath.Path(
    np.c_[np.sin(np.linspace(0, 2 * np.pi, 200)),
          np.cos(np.linspace(0, 2 * np.pi, 200))] * 0.5 + 0.5
)

def event_axes(fig, spec, lat0, lon0, radius=DISC, title=None):
    "One circular panel centred on the event, with coastlines."
    ax = fig.add_subplot(
        spec,
        projection=ccrs.AzimuthalEquidistant(
            central_longitude=float(lon0), central_latitude=float(lat0)
        ),
    )
    ax.set_boundary(_CIRCLE, transform=ax.transAxes)
    metres = radius * 111.195e3
    ax.set_xlim(-metres, metres)
    ax.set_ylim(-metres, metres)
    ax.coastlines(linewidth=0.45, color="0.3")
    ax.gridlines(linewidth=0.3, color="0.6", alpha=0.7,
                 ylocs=range(0, 91, 15), xlocs=range(-180, 181, 30))
    if title:
        ax.set_title(title, fontsize=9)
    return ax

def shade(ax, field, lat, lon, **kwargs):
    "Colour a field on such a panel, closing the longitude seam."
    return ax.pcolormesh(
        np.append(lon, 360.0), lat, np.c_[field, field[:, :1]],
        transform=ccrs.PlateCarree(), shading="auto", **kwargs
    )

def fill(ax, field, lat, lon, levels, **kwargs):
    "The same, as filled contours, for fields with a clear level structure."
    return ax.contourf(
        np.append(lon, 360.0), lat, np.c_[field, field[:, :1]], levels,
        transform=ccrs.PlateCarree(), extend="both", **kwargs
    )

def arrows(ax, u, v, lat, lon, scale=300.0):
    return ax.quiver(lon, lat, u, v, transform=ccrs.PlateCarree(),
                     regrid_shape=18, scale=scale, width=0.005, color="0.15")

def mark(ax, lat0, lon0):
    ax.plot(lon0, lat0, "c*", markersize=13, markeredgecolor="k",
            markeredgewidth=0.7, transform=ccrs.PlateCarree(), zorder=6)
    ax.plot(0.0, 90.0, "o", color="white", markersize=5, markeredgecolor="k",
            markeredgewidth=0.8, transform=ccrs.PlateCarree(), zorder=6)
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. The event and its climatology

The inversion works on an anomaly, so it needs two states: the event and the
climatology for the same day and hour. Both cover the whole northern hemisphere,
because a global inversion has no lateral boundary to stop at.
""")

code(r"""
event_ds = xr.open_dataset(DATA / f"{CASE}.nc")
clim_ds = xr.open_dataset(DATA / f"{CASE}_clim.nc")
centre = json.loads((DATA / "events.json").read_text())[CASE]
lat0, lon0 = centre["lat"], centre["lon"]

lat = event_ds["lat"].values          # ascending, 0 to 90 N
lon = event_ds["lon"].values          # ascending, from 0
plev = event_ds["level"].values       # 1000 hPa first

print(f"{CASE}: {centre['time']}, centre {lat0:.1f} N {lon0:.1f} E")
print(f"grid {lat.size} x {lon.size}, levels {plev.astype(int)}")

def fields(ds):
    "Height, temperature and the two wind components, bottom-up."
    return tuple(np.asarray(ds[v].values, float) for v in ("z", "t", "u", "v"))

z_ev, t_ev, u_ev, v_ev = fields(event_ds)
z_cl, t_cl, u_cl, v_cl = fields(clim_ds)
k250 = int(np.where(plev == TARGET)[0][0])
k500 = int(np.where(plev == 500)[0][0])
""")

code(r"""
anom = z_ev[k500] - z_cl[k500]
fig = plt.figure(figsize=(10, 4.4), constrained_layout=True)
grid = fig.add_gridspec(1, 2)

ax = event_axes(fig, grid[0, 0], lat0, lon0,
                title=f"{TARGET:.0f} hPa wind speed, event")
m = shade(ax, np.hypot(u_ev[k250], v_ev[k250]), lat, lon, cmap="viridis")
mark(ax, lat0, lon0)
fig.colorbar(m, ax=ax, shrink=0.75, label="m s$^{-1}$")

ax = event_axes(fig, grid[0, 1], lat0, lon0, title="500 hPa height anomaly")
lim = float(np.abs(anom).max())
m = shade(ax, anom, lat, lon, cmap="RdBu_r", vmin=-lim, vmax=lim)
mark(ax, lat0, lon0)
fig.colorbar(m, ax=ax, shrink=0.75, label="m")
fig.savefig(FIGURES / "fig01_event.png")
print(f"height anomaly at the centre: "
      f"{anom[np.argmin(abs(lat-lat0)), np.argmin(abs(lon-lon0))]:.0f} m")
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. The vertical coordinate, and potential temperature

The inversion uses the Exner function as its vertical coordinate,

$$\Pi = c_p \left(\frac{p}{p_0}\right)^{\kappa}, \qquad \kappa = R_d/c_p = 2/7,$$

because hydrostatic balance is then simply $\partial\Phi/\partial\Pi = -\theta$,
with no coefficient in the way. $\Pi$ *decreases* upward, so every vertical
difference below carries that sign.

Potential temperature is formed here, from temperature. Supplying potential
temperature instead would leave the state wrong by one factor of the Exner
function, and it would still look entirely plausible.
""")

code(r"""
levels = build_levels("NL9")
assert np.allclose(levels.p_hpa, plev), "the data must be on the levels the solver uses"

print("pressure [hPa]:", levels.p_hpa.astype(int))
print("Pi [J/kg/K]   :", np.round(levels.pi, 1))
print(f"Pi decreases upward, so every vertical difference carries that sign; "
      f"the spacing runs from {abs(levels.pi[1]-levels.pi[0]):.0f} at the bottom "
      f"to {abs(levels.pi[-1]-levels.pi[-2]):.0f} at the top.")
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. The streamfunction, from the wind

The balance system is written in the streamfunction, so the first thing to do with
the wind is take its rotational part. On the sphere that is one spectral operation:
the vorticity comes from the wind directly, and

$$\nabla^2\psi = \zeta$$

is inverted by dividing each spherical-harmonic coefficient by $-n(n+1)/a^2$.
There is no relaxation and no boundary condition to choose, because there is no
boundary — the only mode without an inverse is the global constant, which carries
no wind.

The solver grid is Gaussian and excludes the poles, so no metric factor is ever
divided by at a singular point.
""")

code(r"""
solver = SphereOps(SHT(gaussian_grid(96, 192), lmax=63))
config = InversionConfig(mirror=MirrorConfig(blend=True))

event = prepare_state(z_ev, t_ev, u_ev, v_ev, lat, lon, levels, solver)
clim = prepare_state(z_cl, t_cl, u_cl, v_cl, lat, lon, levels, solver)

slat, slon = solver.grid.lat, solver.grid.lon
north = slat > 0
psi250 = solver.synth(event.psi_spec[k250])
zeta250 = event.zeta[k250]

fig = plt.figure(figsize=(10, 4.4), constrained_layout=True)
grid = fig.add_gridspec(1, 2)
ax = event_axes(fig, grid[0, 0], lat0, lon0, title=r"$\psi$ at 250 hPa")
m = shade(ax, psi250, slat, slon, cmap="RdBu_r")
mark(ax, lat0, lon0)
fig.colorbar(m, ax=ax, shrink=0.75, label="m$^2$ s$^{-1}$")

ax = event_axes(fig, grid[0, 1], lat0, lon0,
                title=r"$\zeta = \nabla^2\psi$ at 250 hPa")
lim = float(np.abs(zeta250[north]).max())
m = shade(ax, zeta250, slat, slon, cmap="RdBu_r", vmin=-lim, vmax=lim)
mark(ax, lat0, lon0)
fig.colorbar(m, ax=ax, shrink=0.75, label="s$^{-1}$")
fig.savefig(FIGURES / "fig02_streamfunction.png")

u_rot, v_rot = rotational_wind_stack(solver, event.psi_spec)
print(f"rotational wind at 250 hPa: rms {np.sqrt(np.mean(u_rot[k250][north]**2 + v_rot[k250][north]**2)):.1f} m/s")
print(f"full wind at 250 hPa      : rms {np.sqrt(np.mean(u_ev[k250]**2 + v_ev[k250]**2)):.1f} m/s")

# The boundary temperature the geopotential implies hydrostatically, over the one
# the temperature gives.  Real data sit near one; a potential temperature handed
# in as a temperature, or a height as a geopotential, does not, and would pass
# every other check.  The pipeline refuses inputs outside 0.7 to 1.4 at the top.
for label, state in (("event", event), ("climatology", clim)):
    bot, top = state.boundary_theta_ratio
    print(f"hydrostatic consistency, {label:12s}: theta ratio bottom {bot:.3f}, top {top:.3f}")
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. Ertel potential vorticity

The quantity being inverted is the hydrostatic Ertel potential vorticity,

$$q = -g\left[(f+\zeta)\frac{\partial\theta}{\partial p}
      + \frac{\partial u}{\partial p}\frac{\partial\theta}{\partial y}
      - \frac{\partial v}{\partial p}\frac{\partial\theta}{\partial x}\right],$$

which the solver carries in the form the balance equations need,

$$\hat q = (f+\nabla^2\psi)\frac{\partial^2\Phi}{\partial\Pi^2}
         - \nabla_h\!\left(\frac{\partial\psi}{\partial\Pi}\right)\cdot
           \nabla_h\!\left(\frac{\partial\Phi}{\partial\Pi}\right).$$

The two differ by a single factor per level, so what is plotted below is in the
familiar potential vorticity units.

Interior potential vorticity exists on levels 850 to 200 hPa only. The vertical
derivative needs a level on each side, so at 1000 and 100 hPa there is none — those
two levels carry **boundary potential temperature** instead, which enters the
inversion as a condition rather than as a source.

Both are evaluated with the inversion's own stencils (`pv_source="operator"`,
the default): the stratification is the three-point second difference of the
geopotential on the stretched Exner grid, the shear terms use the hydrostatic
ghost levels, and the boundary temperature is the hydrostatic difference of the
geopotential across the half level. The data pair then satisfies the
potential-vorticity row of the system exactly, and the balanced state has
nothing to absorb but the imbalance itself. The limited-area convention of plain
centred differences of temperature and wind (`pv_source="data"`) is still
available, and differs from it level by level on a stretched grid.
""")

code(r"""
pv = ertel_pv_pvu(event.q_hat, levels)
interior = levels.interior
kq = list(interior).index(k250)

fig = plt.figure(figsize=(11, 4.4), constrained_layout=True)
grid = fig.add_gridspec(1, 2)
ax = event_axes(fig, grid[0, 0], lat0, lon0, title=f"Ertel PV at {TARGET:.0f} hPa")
m = fill(ax, pv[kq], slat, slon, np.arange(0.0, 8.1, 0.5), cmap="viridis")
mark(ax, lat0, lon0)
fig.colorbar(m, ax=ax, shrink=0.75, label="PVU")

axes = [ax, fig.add_subplot(grid[0, 1])]
i0 = np.argmin(abs(slon - lon0))
section = np.array([pv[i][:, i0] for i in range(len(interior))])
m = axes[1].contourf(slat[north], levels.p_hpa[interior], section[:, north],
                     np.linspace(0, 4, 17), cmap="viridis", extend="both")
axes[1].invert_yaxis(); axes[1].set_yscale("log")
axes[1].set_yticks([850, 700, 500, 400, 300, 250, 200])
axes[1].set_yticklabels([850, 700, 500, 400, 300, 250, 200])
axes[1].minorticks_off()
axes[1].set_xlabel("latitude [deg N]"); axes[1].set_ylabel("pressure [hPa]")
axes[1].set_title(f"section through {lon0:.0f} E")
axes[1].axvline(lat0, color="k", ls=":")
fig.colorbar(m, ax=axes[1], label="PVU")
fig.savefig(FIGURES / "fig03_pv.png")
print("levels carrying interior PV:", levels.p_hpa[interior].astype(int))
print("levels carrying boundary theta:", [int(levels.p_hpa[0]), int(levels.p_hpa[-1])])

# How much the two conventions for the source differ on this grid: the
# operator's stencils against plain centred differences of temperature and wind.
event_data = prepare_state(z_ev, t_ev, u_ev, v_ev, lat, lon, levels, solver, pv_source="data")
pv_data = ertel_pv_pvu(event_data.q_hat, levels)
print("operator vs data source, rms difference over rms PV, northern hemisphere:")
print("  " + ", ".join(
    f"{int(levels.p_hpa[k])}: {100 * np.sqrt(np.mean((pv[i] - pv_data[i])[north] ** 2)) / np.sqrt(np.mean(pv[i][north] ** 2)):.0f}%"
    for i, k in enumerate(interior)))
print(f"  boundary theta, rms difference: bottom {np.sqrt(np.mean((event.theta_bot - event_data.theta_bot)[north] ** 2)):.1f} K, "
      f"top {np.sqrt(np.mean((event.theta_top - event_data.theta_top)[north] ** 2)):.1f} K")
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. The anomaly to be inverted

Subtracting the climatology gives the sources: an interior potential vorticity
anomaly on each of the seven interior levels, and a potential temperature anomaly
on each of the two boundaries. Those nine things are the pieces. Nothing else
drives the inversion.
""")

code(r"""
q_anom = ertel_pv_pvu(event.q_hat - clim.q_hat, levels)
theta_bot_anom = event.theta_bot - clim.theta_bot
theta_top_anom = event.theta_top - clim.theta_top

# The nine sources are plotted further down, each beside the wind it induces,
# which is where they mean something.
print(f"PV anomaly at {TARGET:.0f} hPa: min {q_anom[kq][north].min():.2f}, max {q_anom[kq][north].max():.2f} PVU")
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. Extending the hemisphere to the sphere

The data are northern-hemisphere only; the operator is global. The extension has
to keep the inversion elliptic and must not let the invented hemisphere leak into
the answer, and one choice does both: reflect everything as an **even** function of
latitude and use $|f|$.

With even coefficients and even sources the operator commutes with a reflection
about the equator, so the solution is exactly even, and its northern half solves
the northern problem under a condition of no meridional gradient at the equator.
The southern half is then determined by the northern one and carries no
information of its own.

Two details matter. The wind is mirrored **as a vector** first — eastward even,
northward odd — and its vorticity taken; only then is that vorticity mirrored as
an even scalar. Mirroring the wind and stopping there would leave $\zeta$ odd, so
$|f| + \zeta$ would change sign across the equator and ellipticity would be lost.
And $|f|$ has a corner at the equator, which is smoothed by a floor.

The mirrored reference state has a kink at the equator too, and the coefficients
built from it are tapered over a band of latitude: the weight in the right-hand
panel below multiplies the *products* the quadratic terms of the operator are
made of, never the sources or the solution. Written that way the tapered system
is still an exact quadratic with a symmetric bilinear form, so the midpoint
linearisation of the next section holds with the taper on. Tapering the reference
vorticity alone, which is what an earlier version did, made the operator the
tangent of two different flows at once inside the band.
""")

code(r"""
from pvinv_sph.mirror import blend_weight, coriolis_star

fig, axes = plt.subplots(1, 3, figsize=(14, 3.2), constrained_layout=True)
axes[0].pcolormesh(slon, slat, solver.synth(event.psi_spec[k250]), cmap="RdBu_r",
                   shading="auto")
axes[0].axhline(0, color="k", lw=0.8)
try:                                    # coastlines, drawn straight on the axes
    import cartopy.feature as cfeature
    for geom in cfeature.COASTLINE.geometries():
        for line in (geom.geoms if hasattr(geom, "geoms") else [geom]):
            xy = np.asarray(line.coords)
            axes[0].plot(xy[:, 0] % 360.0, xy[:, 1], color="0.35", lw=0.3)
except Exception:
    pass
axes[0].set_title(r"$\psi$ on the sphere: the southern half is the mirror")
axes[0].set_xlabel("longitude [deg E]"); axes[0].set_ylabel("latitude [deg N]")

deg = np.linspace(0, 90, 400)
axes[1].plot(deg, coriolis_star(deg, 12.0) / (2 * 7.292115e-5 * np.sin(np.radians(np.maximum(deg, 1e-6)))))
axes[1].set_xlim(0, 60); axes[1].set_ylim(1, 3)
axes[1].set_xlabel("latitude [deg N]"); axes[1].set_ylabel(r"$f^*/|f|$")
axes[1].set_title("the Coriolis floor, 12 deg")
axes[1].grid(alpha=0.3)

axes[2].plot(deg, blend_weight(deg, 5.0, 20.0))
axes[2].set_xlim(0, 40)
axes[2].set_xlabel("latitude [deg N]"); axes[2].set_ylabel("weight")
axes[2].set_title("the equatorial taper of the coefficients")
axes[2].grid(alpha=0.3)
fig.savefig(FIGURES / "fig04_mirror.png")

for d in (10.5, 20.0, 30.0, 45.0):
    r = float(coriolis_star(np.array([d]), 12.0)[0]) / (2 * 7.292115e-5 * np.sin(np.radians(d)))
    print(f"  {d:5.1f} N: f*/|f| = {r:.2f}, taper weight = {float(blend_weight(np.array([d]))[0]):.2f}")
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. The solve

Two equations are solved together for the perturbation geopotential and
streamfunction: the balance equation, and the potential vorticity equation
linearised about the mean plus half the perturbation. That midpoint is not an
approximation — for a quadratic system it is exact — and it is what makes the
pieces add up.

The total balanced state is found first, by a Newton iteration whose Jacobian is
the same operator the pieces use. Each linear solve is preconditioned by a
separable approximation: the horizontal part is diagonal in spherical-harmonic
space and the vertical part is a small tridiagonal system for each coefficient.

The linearised balance equation is elliptic only where the absolute vorticity of
the reference flow exceeds its deformation. On the flank of a strong anticyclone
it does not, and a Newton iteration that reaches such a state walks into a fold.
There is a safety net for that case: the deformation part of the balance row
can be limited by $s = \min\!\left(1,\ (1-m)\,\mathrm{AVO}/(w D)\right)$
with a margin $m = 0.05$, the vorticity part, which is what gradient-wind
balance lives on, being kept. By default the limiter is brought in adaptively:
the iteration starts on the balance equation as posed and only if an inner
solve fails to converge or a line search fails — the two signs that the
linearised row has lost ellipticity — is the limiter switched on, from the
observed state and the iterate, and the step retaken. While it does not
change, the residual is one quadratic system and every Newton step is exact;
the pieces take whatever limiter the total converged with, so their operator
is the tangent of the same system. Convergence is declared when the
geopotential increment falls below 0.1 m after a successful line search — and
the report also carries the residuals of the equations *as posed*, without
taper, floors or limiter, so "converged" is a number in metres and in PVU
rather than a flag.

On this event the iteration takes five steps, with increments of 308, 21.6,
1.50, 0.12 and 0.04 m, where the earlier construction, with a limiter that
followed the iterate and a taper on the reference alone, took eleven. No inner
solve stops short of its tolerance and the limiter is never needed, although
the linearised balance row is not elliptic at the returned state on 1.5 % of
the area at 850 hPa rising to 14 % at 300 hPa — the strain-dominated flanks
of the jet: an indefinite region that small is something the preconditioned
Krylov solves and the line search get through on their own. The posed
potential-vorticity equation is met to 0.0040 PVU rms poleward of 20 N.
""")

code(r"""
import time

started = time.time()
result = invert_pieces(solver, levels, clim, event, cfg=config)
elapsed = time.time() - started

report = result.diagnostics
print(f"Newton: {result.newton_steps} steps, converged {report['newton_converged']}, "
      f"final increment {report['newton_final_increment_m']:.3f} m")
print(f"Newton increments [m]: {np.round(np.array(report['newton_increments']) / 9.81, 3)}")
print(f"Krylov iterations per piece: {report['linear_iterations']}")
print(f"inner solves short of tolerance: {report['inner_solves_unconverged']}")
print(f"clamped fraction: {result.clamp_worst:.4f}")
print(f"elapsed: {elapsed:.0f} s")

# What "converged" means physically: the residuals of the two equations as posed
# (no taper, no floors, no limiter) at the balanced state, in metres of height
# for the balance row and in PVU for the potential-vorticity row.  Poleward of
# the taper band the solved system and the posed one coincide, so those are
# the numbers to read; the global maxima include the band, where they differ
# by construction.
norms = report["newton_final_norms"]
print(f"posed-equation residuals poleward of {config.mirror.blend_north:.0f} N: "
      f"balance max {norms['balance_m_extratropics']:.3f} m, "
      f"PV max {norms['pv_pvu_extratropics']:.4f} PVU, "
      f"PV rms {norms['pv_pvu_rms_extratropics']:.5f} PVU")
print(f"posed-equation residuals, whole sphere:      "
      f"balance max {norms['balance_m']:.3f} m, PV max {norms['pv_pvu']:.4f} PVU")

# Where the balance row loses ellipticity -- the reference deformation exceeding
# the absolute vorticity -- the deformation terms can be limited.  The area
# fraction on which the limiter acted, per interior level, for the total and for
# the pieces (which share it); how often it was brought in; and the fraction of
# each level where the linearised balance row is still not elliptic at the
# returned state.
def per_level(values):
    return ", ".join(f"{int(levels.p_hpa[k])}:{100 * v:.1f}%" for k, v in zip(interior, values))
print("deformation limiter active (total): ", per_level(report["newton_deformation_fraction"]))
print("deformation limiter active (pieces):", per_level(report["piece_deformation_fraction"]))
print(f"limiter refreshes: {report['newton_limiter_refreshes']}")
print("non-elliptic at the returned state:  ", per_level(report["newton_final_nonelliptic_fraction"]))

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), constrained_layout=True)
axes[0].semilogy(np.arange(1, len(report["newton_increments"]) + 1),
                 np.array(report["newton_increments"]) / 9.81, "o-")
axes[0].axhline(0.1, color="r", ls="--", label="tolerance, 0.1 m")
axes[0].set_xticks(np.arange(1, len(report["newton_increments"]) + 1))
axes[0].set_xlabel("Newton step"); axes[0].set_ylabel("increment [m]")
axes[0].set_title("the nonlinear iteration"); axes[0].legend(); axes[0].grid(alpha=0.3)

names = list(report["linear_iterations"])
axes[1].bar(range(len(names)), [report["linear_iterations"][n] for n in names])
axes[1].set_xticks(range(len(names))); axes[1].set_xticklabels(names, rotation=45, ha="right")
axes[1].set_ylabel("Krylov iterations"); axes[1].set_title("one linear solve per piece")
axes[1].grid(alpha=0.3, axis="y")
fig.savefig(FIGURES / "fig05_convergence.png")
""")

md(r"""
The balance residual above has a large maximum, and it is not where it seems.
The balance row is reported as a height by inverting a Laplacian, which is a
global operation: the residual inside the taper band, where the solved system
differs from the posed one by construction, spreads into every latitude of the
height field, and a mask applied afterwards does not remove it. The cell below
measures the same residual locally, as the vorticity error it is equivalent to
(the residual over $f$), separately in the strain-dominated region of the
observed state — where the limiter would act if it were needed — and outside
it, with the observed state's own imbalance and the vorticity itself for
scale; and it inverts the Laplacian of each region's residual on its own.
""")

code(r"""
from pvinv_sph.levels import G
from pvinv_sph.operator import FrozenState
from pvinv_sph.passc import BalancedInversion
from pvinv_sph.qmin import floor_pv

balancer = BalancedInversion(solver, levels, clamps=config.clamps, mirror=config.mirror,
                             krylov=config.krylov, newton=config.newton)
q_total, _ = floor_pv(event.q_hat, levels, solver.grid.weights,
                      config.pv_floor.qmin_total, config.clamps.mode)

def balance_residual_grid(psi_spec, phi_spec):
    "The posed balance row at a state, on the grid, in s^-2."
    r1, _ = balancer.residual(psi_spec, phi_spec, q_total, event.theta_bot, event.theta_top)
    return np.stack([solver.synth(r1[k]) for k in range(r1.shape[0])])

frozen_obs = FrozenState(solver, levels, event.psi_spec, event.phi_spec,
                         clamps=config.clamps, mirror=config.mirror)
limit, f_grid = frozen_obs.deform_limit, frozen_obs.f_grid
r_bal = balance_residual_grid(result.total_psi_spec, result.total_phi_spec) / f_grid
r_obs = balance_residual_grid(event.psi_spec, event.phi_spec) / f_grid
zeta_bal = np.stack([solver.synth(solver.lap(result.total_psi_spec[k])) for k in interior])
outside = np.broadcast_to((np.abs(slat) >= config.mirror.blend_north)[:, None], r_bal.shape[1:])

def rms_on(field, mask):
    return float(np.sqrt(np.mean(field[mask] ** 2))) if mask.any() else float("nan")

def height_of(field, mask):
    "Invert the Laplacian of the residual restricted to one region; rms poleward of the band, in m."
    h = solver.synth(solver.inv_lap(solver.analyze(np.where(mask, field, 0.0)))) / G
    return rms_on(h, outside)

print("posed balance residual as an equivalent vorticity error [s^-1], poleward of the taper band")
print("  level   balanced, quiet   balanced, strain   observed state   vorticity rms   |  as height [m]: quiet region   strain region")
for i, k in enumerate(interior):
    quiet, acted = (limit[i] >= 1.0) & outside, (limit[i] < 1.0) & outside
    print(f"  {int(levels.p_hpa[k]):5d}   {rms_on(r_bal[i], quiet):15.2e}   {rms_on(r_bal[i], acted):15.2e}   "
          f"{rms_on(r_obs[i], outside):14.2e}   {rms_on(zeta_bal[i], outside):13.2e}   |  "
          f"{height_of(r_bal[i] * f_grid, quiet):19.2f}   {height_of(r_bal[i] * f_grid, acted):12.2f}")
band = ~outside
print(f"inside the taper band, all levels: balanced {np.sqrt(np.mean(r_bal[:, band] ** 2)):.2e} s^-1 rms, "
      f"observed {np.sqrt(np.mean(r_obs[:, band] ** 2)):.2e}; the maximum quoted above lives here")
""")

md(r"""
Read locally, the balanced state satisfies the posed balance equation poleward
of the taper band to 1–4 × 10⁻⁷ s⁻¹, in the strain-dominated region as much as
outside it: two orders of magnitude below the observed state's own imbalance
of 1.4–2.5 × 10⁻⁵ s⁻¹, and half a percent of the vorticity itself. Inverted
region by region, the residual is worth 0.2–3 m of height outside the strain
region and 0.04–0.8 m inside it. Inside the taper band the residual is
4.6 × 10⁻⁶ s⁻¹, and that is where the 25 m maximum of the report comes from:
a height norm masked after the inverse Laplacian is not an extratropical
norm. The potential-vorticity norms of the report have no such step and can be
read as they are.
""")

# ---------------------------------------------------------------------------
md(r"""
## 8. The wind induced by each level

This is the result. Each panel is the rotational wind at 250 hPa that would exist
if the only anomaly in the atmosphere were the one on that level, with everything
else at its climatological value.

The two boundary panels are potential temperature anomalies; the other seven are
potential vorticity anomalies.

The numbers in the titles are the rms of each induced wind over the hemisphere.
For this event they climb with height: 3.83 m/s from the surface potential
temperature, about 2 m/s from each of 850, 700 and 500 hPa, 3.19 at 400,
7.21 at 300, 10.29 at 250 and 11.73 at 200 hPa — and 14.66 m/s from the 100 hPa
boundary potential temperature, the largest single piece. That top boundary
sits 5 km above the level being explained and carries the anomaly of the
lower-stratospheric temperature, which for a block is large; how much of it is
"the block" and how much is the lid of the domain is a question the grouping in
the next section leaves open by putting it with the upper levels.
""")

code(r"""
def induced(piece_name):
    "Rotational wind at the target level induced by one piece."
    u, v = rotational_wind_stack(solver, result.pieces[piece_name].psi_spec)
    return u[k250], v[k250]

# Each panel pairs a source with what it induces: the shading is the anomaly that
# drives the piece -- potential vorticity on the seven interior levels, potential
# temperature on the two boundaries -- and the arrows are the wind it produces at
# the target level.  Those are different quantities in different units, so the
# arrow scale is set panel by panel and written on each one; comparing arrow
# lengths between panels means reading the keys.
order = [str(int(p)) for p in levels.p_hpa]
sources = {order[0]: (theta_bot_anom, 6.0, "K"),
           order[-1]: (theta_top_anom, 12.0, "K")}
q_lim = float(np.percentile(np.abs(q_anom[:, north, :]), 99))
for i, k in enumerate(interior):
    sources[str(int(levels.p_hpa[k]))] = (q_anom[i], q_lim, "PVU")

fig = plt.figure(figsize=(13.5, 12.5), constrained_layout=True)
grid = fig.add_gridspec(3, 3)
for n, name in enumerate(order):
    u, v = induced(name)
    field, vlim, unit = sources[name]
    speed = np.hypot(u, v)
    rms = float(np.sqrt(np.mean(speed[north] ** 2)))
    kind = "boundary theta anomaly" if unit == "K" else "PV anomaly"
    ax = event_axes(fig, grid[n // 3, n % 3], lat0, lon0,
                    title=f"{name} hPa {kind}   induced rms {rms:.1f} m s$^{{-1}}$")
    m = shade(ax, field, slat, slon, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    # One arrow length per panel, scaled to that panel's own strength, so a weak
    # piece stays legible instead of vanishing beside a strong one.
    key = max(1.0, float(np.round(2 * rms)))
    handle = arrows(ax, u, v, slat, slon, scale=18.0 * key)
    ax.quiverkey(handle, 0.78, 0.05, key, f"{key:.0f} m s$^{{-1}}$",
                 labelpos="E", coordinates="axes", fontproperties={"size": 8})
    mark(ax, lat0, lon0)
    fig.colorbar(m, ax=ax, shrink=0.72, label=unit)
    print(f"{name:>5s} hPa {kind:22s} induced rms {rms:5.2f} m/s")
fig.suptitle(f"each level's anomaly (shading) and the {TARGET:.0f} hPa rotational "
             f"wind it induces (arrows)")
fig.savefig(FIGURES / "fig06_per_level.png")
""")

# ---------------------------------------------------------------------------
md(r"""
## 9. Grouped into surface, lower and upper

Nine panels is more than the eye can hold, and the usual question is coarser: how
much of the upper-level flow came from below, and how much from the upper
troposphere? The pieces are linear in their sources, so they can simply be added:

* **surface** — the 1000 hPa boundary potential temperature
* **lower** — potential vorticity at 850, 700 and 500 hPa
* **upper** — potential vorticity at 400, 300, 250 and 200 hPa, together with the
  100 hPa boundary potential temperature

The three groups exhaust the sources, so they sum to the same thing the nine
panels do. For this event the upper group dominates: 20.68 m/s rms against
3.83 from the surface and 3.68 from the lower troposphere, beside an observed
anomaly of 21.09 m/s.
""")

code(r"""
GROUPS = {
    "surface": ["1000"],
    "lower": ["850", "700", "500"],
    "upper": ["400", "300", "250", "200", "100"],
}

def group_wind(names):
    total = sum(result.pieces[n].psi_spec for n in names)
    u, v = rotational_wind_stack(solver, total)
    return u[k250], v[k250]

u_obs, v_obs = rotational_wind_stack(solver, event.psi_spec - clim.psi_spec)
u_obs, v_obs = u_obs[k250], v_obs[k250]

panels = [(name, *group_wind(names)) for name, names in GROUPS.items()]
panels.append(("observed anomaly", u_obs, v_obs))
fig = plt.figure(figsize=(17, 4.6), constrained_layout=True)
grid = fig.add_gridspec(1, 4)
for col, (name, u, v) in enumerate(panels):
    speed = np.hypot(u, v)
    rms = float(np.sqrt(np.mean(speed[north] ** 2)))
    ax = event_axes(fig, grid[0, col], lat0, lon0,
                    title=f"{name}   rms {rms:.1f} m s$^{{-1}}$")
    # Colour and arrows are both scaled to the panel's own strength.  The groups
    # differ by a factor of more than five here, and one shared scale would leave
    # two of the four blank; the numbers in the titles are what to compare.
    vmax = float(np.percentile(speed[north], 98))
    m = shade(ax, speed, slat, slon, cmap="magma_r", vmin=0, vmax=vmax)
    key = max(1.0, float(np.round(2 * rms)))
    handle = arrows(ax, u, v, slat, slon, scale=18.0 * key)
    ax.quiverkey(handle, 0.78, 0.05, key, f"{key:.0f} m s$^{{-1}}$",
                 labelpos="E", coordinates="axes", fontproperties={"size": 8})
    mark(ax, lat0, lon0)
    fig.colorbar(m, ax=ax, shrink=0.72, label="m s$^{-1}$")
    print(f"{name:17s} rms {rms:5.2f} m/s")
fig.suptitle(f"rotational wind at {TARGET:.0f} hPa, grouped by the depth of the source")
fig.savefig(FIGURES / "fig07_groups.png")
""")

# ---------------------------------------------------------------------------
md(r"""
## 10. What the pieces add up to

Two checks close the calculation.

The first is arithmetic: the nine pieces must reproduce the inversion of all the
sources at once. They are solved separately against one frozen operator, and the
system is linear in the sources, so this holds to the solver's tolerance and not
merely approximately. If it did not, the decomposition would mean nothing.

The second is physical: the sum of the pieces against the observed anomalous
rotational wind. What is left over is the part the balance equations do not
represent — the divergent and unbalanced flow. It is not an error, and its size
is worth knowing. For this event the pieces reproduce the all-sources inversion
to 1.7 parts in 10⁸, sum to 20.01 m/s rms against an observed anomaly of
21.09 m/s, and leave 3.10 m/s, 15 % of the anomaly. Seen from the other side,
the balanced total state misses the observed rotational wind by 4.97 m/s rms
out of 30.99, and the pieces miss the balanced perturbation by 4.14 m/s: the
first is what the balance equations do not carry, the second is what the
midpoint linearisation costs where the floors act and, since the mean state
enters as observed rather than balanced, the imbalance of the mean.
""")

code(r"""
from pvinv_sph.passd import all_sources_inversion

summed = result.summed_psi()
total, total_report = all_sources_inversion(solver, levels, clim, event, cfg=config)
closure = np.abs(summed - total).max() / np.abs(total).max()
print(f"sum of pieces vs all sources at once: {closure:.2e}  (converged {total_report.converged})")

u_sum, v_sum = rotational_wind_stack(solver, summed)
u_sum, v_sum = u_sum[k250], v_sum[k250]
residual_u, residual_v = u_obs - u_sum, v_obs - v_sum

def rms(a, b):
    return float(np.sqrt(np.mean(a[north] ** 2 + b[north] ** 2)))

print(f"observed anomaly     {rms(u_obs, v_obs):6.2f} m/s")
print(f"sum of the pieces    {rms(u_sum, v_sum):6.2f} m/s")
print(f"left over            {rms(residual_u, residual_v):6.2f} m/s"
      f"  ({100 * rms(residual_u, residual_v) / rms(u_obs, v_obs):.0f} % of the anomaly)")

# The same residual seen from the other side.  The balanced total state is the
# nonlinear inversion of all the event's sources; its rotational wind against the
# observed one is the part of the flow the balance equations do not carry, and
# the pieces against the balanced perturbation is what the midpoint
# linearisation costs (the piece sum reproduces the linearised total exactly,
# the balanced perturbation only up to the floors and the limiter).
u_bal, v_bal = rotational_wind_stack(solver, result.total_psi_spec)
u_bp, v_bp = rotational_wind_stack(solver, result.psi_perturbation)
u_full, v_full = rotational_wind_stack(solver, event.psi_spec)
print(f"balanced total vs observed rotational wind   {rms(u_full[k250] - u_bal[k250], v_full[k250] - v_bal[k250]):6.2f} m/s"
      f"  (observed rotational wind rms {rms(u_full[k250], v_full[k250]):.2f} m/s)")
print(f"sum of the pieces vs balanced perturbation   {rms(u_sum - u_bp[k250], v_sum - v_bp[k250]):6.2f} m/s")

fig = plt.figure(figsize=(14, 4.6), constrained_layout=True)
grid = fig.add_gridspec(1, 3)
vmax = float(np.percentile(np.hypot(u_obs, v_obs)[north], 98))
for col, (name, u, v) in enumerate([("observed anomaly", u_obs, v_obs),
                                    ("sum of the pieces", u_sum, v_sum),
                                    ("left over", residual_u, residual_v)]):
    speed = np.hypot(u, v)
    rms = float(np.sqrt(np.mean(speed[north] ** 2)))
    ax = event_axes(fig, grid[0, col], lat0, lon0,
                    title=f"{name}   rms {rms:.1f} m s$^{{-1}}$")
    m = shade(ax, speed, slat, slon, cmap="magma_r", vmin=0, vmax=vmax)
    key = max(1.0, float(np.round(2 * rms)))
    handle = arrows(ax, u, v, slat, slon, scale=18.0 * key)
    ax.quiverkey(handle, 0.78, 0.05, key, f"{key:.0f} m s$^{{-1}}$",
                 labelpos="E", coordinates="axes", fontproperties={"size": 8})
    mark(ax, lat0, lon0)
    fig.colorbar(m, ax=ax, shrink=0.72, label="m s$^{-1}$")
fig.savefig(FIGURES / "fig08_closure.png")
""")

# ---------------------------------------------------------------------------
md(r"""
## 11. The same calculation at 79 N

Nothing above referred to where the event was. The Greenland case that ships with
the package sits at 79 N, where half of a box drawn around it would lie across the
pole, and the calculation is the same one.
""")

code(r"""
polar_ev = xr.open_dataset(DATA / "cesm_greenland.nc")
polar_cl = xr.open_dataset(DATA / "cesm_greenland_clim.nc")
polar_centre = json.loads((DATA / "events.json").read_text())["cesm_greenland"]

plat, plon = polar_ev["lat"].values, polar_ev["lon"].values
polar_solver = SphereOps(SHT(gaussian_grid(96, 192), lmax=63))
p_event = prepare_state(*fields(polar_ev), plat, plon, levels, polar_solver)
p_clim = prepare_state(*fields(polar_cl), plat, plon, levels, polar_solver)
started = time.time()
p_result = invert_pieces(polar_solver, levels, p_clim, p_event, cfg=config)
p_report = p_result.diagnostics
print(f"Newton: {p_result.newton_steps} steps, "
      f"converged {p_report['newton_converged']}, "
      f"final increment {p_report['newton_final_increment_m']:.3f} m, "
      f"elapsed {time.time() - started:.0f} s")
p_norms = p_report["newton_final_norms"]
print(f"posed-equation residuals poleward of {config.mirror.blend_north:.0f} N: "
      f"balance max {p_norms['balance_m_extratropics']:.3f} m, "
      f"PV rms {p_norms['pv_pvu_rms_extratropics']:.5f} PVU")
print("deformation limiter active (total): " +
      ", ".join(f"{int(levels.p_hpa[k])}:{100 * v:.1f}%"
                for k, v in zip(interior, p_report["newton_deformation_fraction"])))

p_summed = p_result.summed_psi()
p_total, p_total_report = all_sources_inversion(polar_solver, levels, p_clim, p_event, cfg=config)
print(f"sum of pieces vs all sources at once: "
      f"{np.abs(p_summed - p_total).max() / np.abs(p_total).max():.2e}")
u_psum, v_psum = rotational_wind_stack(polar_solver, p_summed)
u_pobs, v_pobs = rotational_wind_stack(polar_solver, p_event.psi_spec - p_clim.psi_spec)
p_north = polar_solver.grid.lat > 0
def p_rms(a, b):
    return float(np.sqrt(np.mean(a[k250][p_north] ** 2 + b[k250][p_north] ** 2)))
print(f"observed anomaly {p_rms(u_pobs, v_pobs):6.2f} m/s, sum of the pieces {p_rms(u_psum, v_psum):6.2f} m/s, "
      f"left over {p_rms(u_pobs - u_psum, v_pobs - v_psum):6.2f} m/s "
      f"({100 * p_rms(u_pobs - u_psum, v_pobs - v_psum) / p_rms(u_pobs, v_pobs):.0f} % of the anomaly)")

import cartopy.crs as ccrs
import matplotlib.path as mpath

plat_s, plon_s = polar_solver.grid.lat, polar_solver.grid.lon
theta_circle = np.linspace(0, 2 * np.pi, 200)
circle = mpath.Path(np.c_[np.sin(theta_circle), np.cos(theta_circle)] * 0.5 + 0.5)
projection = ccrs.AzimuthalEquidistant(central_longitude=polar_centre["lon"],
                                       central_latitude=polar_centre["lat"])

def polar_group(names):
    total = sum(p_result.pieces[n].psi_spec for n in names)
    u, v = rotational_wind_stack(polar_solver, total)
    return u[k250], v[k250]

u_pobs, v_pobs = rotational_wind_stack(polar_solver, p_event.psi_spec - p_clim.psi_spec)
panels = [(n, *polar_group(names)) for n, names in GROUPS.items()]
panels.append(("observed anomaly", u_pobs[k250], v_pobs[k250]))

fig = plt.figure(figsize=(16, 4.4), constrained_layout=True)
grid = fig.add_gridspec(1, 4)
for col, (name, u, v) in enumerate(panels):
    ax = fig.add_subplot(grid[0, col], projection=projection)
    ax.set_boundary(circle, transform=ax.transAxes)
    radius = 40.0 * 111.195e3
    ax.set_xlim(-radius, radius); ax.set_ylim(-radius, radius)
    speed = np.hypot(u, v)
    vmax = float(np.nanpercentile(speed[plat_s > 0], 98))
    m = ax.pcolormesh(np.append(plon_s, 360.0), plat_s,
                      np.c_[speed, speed[:, :1]], transform=ccrs.PlateCarree(),
                      cmap="magma_r", vmin=0, vmax=vmax, shading="auto")
    rms = float(np.sqrt(np.mean(speed[plat_s > 0] ** 2)))
    key = max(1.0, float(np.round(2 * rms)))
    handle = ax.quiver(plon_s, plat_s, u, v, transform=ccrs.PlateCarree(),
                       regrid_shape=18, scale=18.0 * key, width=0.005, color="0.15")
    ax.quiverkey(handle, 0.78, 0.05, key, f"{key:.0f} m s$^{{-1}}$",
                 labelpos="E", coordinates="axes", fontproperties={"size": 8})
    ax.plot(polar_centre["lon"], polar_centre["lat"], "c*", markersize=14,
            transform=ccrs.PlateCarree())
    ax.plot(0.0, 90.0, "o", color="white", markersize=6, markeredgecolor="k",
            transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.45, color="0.3")
    ax.gridlines(linewidth=0.3, color="0.6", alpha=0.7)
    ax.set_title(name)
    fig.colorbar(m, ax=ax, shrink=0.75, label="m s$^{-1}$")
fig.suptitle(f"{TARGET:.0f} hPa induced wind, Greenland case at "
             f"{polar_centre['lat']:.0f} N; the white circle is the pole")
fig.savefig(FIGURES / "fig09_polar.png")
""")

md(r"""
The pole sits inside every panel and there is nothing special about it. That is
the whole point of doing the inversion globally: the method does not know where
the event is. The Greenland case converges in five Newton steps to a final
increment of 0.003 m, the pieces reproduce the all-sources inversion to
5.1 parts in 10⁹, and they leave 4.17 m/s of a 20.25 m/s anomaly, 21 %,
unattributed.
""")


def main() -> int:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(text) if kind == "markdown"
        else nbf.v4.new_code_cell(text)
        for kind, text in CELLS
    ]
    notebook.metadata["kernelspec"] = {
        "display_name": "blocking",
        "language": "python",
        "name": "blocking",
    }
    path = HERE / "01_walkthrough.ipynb"
    nbf.write(notebook, path)
    print(f"wrote {path} with {len(notebook.cells)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
