"""sh_ppvi — Spherical-harmonic piecewise PV inversion (PPVI).

Python translation of Wu / Davis 1994 Fortran PPVI code, with finite-difference
horizontal operators replaced by spherical-harmonic equivalents (pyspharm).

Passes
------
  pvpialln_sh   (pv_calc.py)  — Pass A/B: compute θ, ζ, ψ, Ertel PV
  balnc_sh      (balnc.py)    — Pass C: total PV inversion (BALNC)
  balp_sh       (balp.py)     — Pass D: piecewise perturbation inversion (BALP)

Supporting modules
------------------
  coords.py    — physical constants, Exner grid, FD matrices
  operators.py — pyspharm wrappers (laplacian, gradient, vortdiv, helmholtz, …)
  io.py        — ERA5 loading and NetCDF output helpers
  sor.py       — SOR solvers (psi_sor_level, H_sor_3d, outer_total_iteration)
"""

from .pv_calc import pvpialln_sh, ertel_pv_sh
from .balnc   import balnc_sh
from .balp    import balp_sh, make_pieces

__all__ = [
    "pvpialln_sh",
    "ertel_pv_sh",
    "balnc_sh",
    "balp_sh",
    "make_pieces",
]
