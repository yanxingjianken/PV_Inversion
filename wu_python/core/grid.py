# wu_python/core/grid.py — Grid metrics and finite-difference coefficients
"""Grid definition and metric arrays for the Wu 51×87 NH sigma-coordinate box.

Ports the grid setup from pvpialln_94UV.f lines ~130-180:
  - Coriolis parameter FC(i) at each latitude
  - cos(lat) factors AP(i), APM(i), APP(i) for map scaling
  - 5-point Laplacian coefficients A(i,1:5)
  - Grid spacings DL, DP in metres

All arrays are indexed [i, j] where i = 0..NY-1 (north→south),
j = 0..NX-1 (west→east), matching the Fortran convention.
"""

import numpy as np
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Grid parameters (from root config)
NX: int = 87
NY: int = 51
NW: int = 10
DLAT: float = 1.5       # degrees
DLON: float = 1.5       # degrees

# Wu domain: 10.5°N to 85.5°N, 169.5°W to 40.5°W
LAT_S: float = 10.5
LAT_N: float = 85.5
LON_W: float = -169.5
LON_E: float = -40.5

# Earth radius [m] and derived grid spacings
AA: float = 2.0e7 / np.pi        # Earth radius
DL: float = AA * np.radians(DLON)  # dx at equator [m]
DP: float = AA * np.radians(DLAT)  # dy [m]
SIGM: float = DLON / DLAT          # map factor ratio (= 1.0 for 1.5°×1.5°)

# Sigma levels (σ = p/1000 hPa)
PR: np.ndarray = np.array([1.0, 0.925, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2])


def build_grid_arrays() -> dict:
    """Build all grid metric arrays matching pvpialln_94UV.f lines ~140-180.

    Returns dict with keys:
        lats, lons     — 1D coordinate arrays
        FC             — Coriolis parameter [s⁻¹]  (NY,)
        AP             — cos(lat) at cell centres   (NY,)
        APM            — cos(lat) at cell S-edges   (NY,)
        APP            — cos(lat) at cell N-edges   (NY,)
        A              — 5-pt Laplacian coeffs      (NY, 5)
        LON2D, LAT2D   — 2D meshgrid for plotting
    """
    lats = LAT_N - np.arange(NY) * DLAT   # north→south
    lons = LON_W + np.arange(NX) * DLON   # west→east

    # Coriolis parameter
    FC = 1.458e-4 * np.sin(np.radians(lats))

    # cos(lat) factors
    AP = np.cos(np.radians(lats))
    APM = np.cos(np.radians(LAT_N - (np.arange(NY) + 0.5) * DLAT))
    APP = np.cos(np.radians(LAT_N - (np.arange(NY) - 0.5) * DLAT))

    # 5-point Laplacian coefficients A(i, 1:5)
    # A(i,1) = SIGM² * APM(i) / AP(i)
    # A(i,2) = 1 / AP(i)²
    # A(i,3) = -(2 + SIGM² * AP(i) * (APM(i) + APP(i))) / AP(i)²
    # A(i,4) = 1 / AP(i)²
    # A(i,5) = SIGM² * APP(i) / AP(i)
    A = np.zeros((NY, 5))
    A[:, 0] = SIGM * SIGM * APM / AP
    A[:, 1] = 1.0 / (AP * AP)
    A[:, 2] = -(2.0 + SIGM * SIGM * AP * (APM + APP)) / (AP * AP)
    A[:, 3] = 1.0 / (AP * AP)
    A[:, 4] = SIGM * SIGM * APP / AP

    LON2D, LAT2D = np.meshgrid(lons, lats)

    return {
        "lats": lats, "lons": lons,
        "FC": FC, "AP": AP, "APM": APM, "APP": APP,
        "A": A, "LON2D": LON2D, "LAT2D": LAT2D,
        "DL": DL, "DP": DP, "SIGM": SIGM, "AA": AA,
    }


# Build at import time for convenience
_grid = build_grid_arrays()
lats = _grid["lats"]
lons = _grid["lons"]
FC = _grid["FC"]
AP = _grid["AP"]
APM = _grid["APM"]
APP = _grid["APP"]
A = _grid["A"]
LON2D = _grid["LON2D"]
LAT2D = _grid["LAT2D"]
