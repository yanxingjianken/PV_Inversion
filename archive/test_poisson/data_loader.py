"""Load ERA5 / Wu-grid data for Poisson solver benchmarks.

Uses the existing pv_inversion data pipeline — reads Wu .grid files
and extracts the fields needed for inversion tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# Ensure wu_python is importable
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from .config import EVENT_GRID, MEAN_GRID


def load_wu_event() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the Wu event-day .grid file.

    Returns:
        ``(H, TH, U, V)`` each shape ``(10, 51, 87)``:
        - *H*:  geopotential height [m]
        - *TH*: potential temperature [K] (Fortran-converted)
        - *U*:  zonal wind [m/s]
        - *V*:  meridional wind [m/s]
    """
    from wu_python.core.io import read_wu_grid
    return read_wu_grid(str(EVENT_GRID))


def load_wu_mean() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the Wu mean-climatology .grid file.

    Returns same shape/semantics as :func:`load_wu_event`.
    """
    from wu_python.core.io import read_wu_grid
    return read_wu_grid(str(MEAN_GRID))


def compute_vorticity_era5(
    U: np.ndarray, V: np.ndarray
) -> np.ndarray:
    """Compute relative vorticity from ERA5 winds (Wu conventions).

    Args:
        U: Zonal wind, shape ``(nlev, nlat, nlon)`` [m/s].
        V: Meridional wind, shape ``(nlev, nlat, nlon)`` [m/s].

    Returns:
        Relative vorticity ζ [s⁻¹], same shape.
    """
    from wu_python.core.pv_calc import compute_relative_vorticity
    return compute_relative_vorticity(U, V)


def compute_pv_era5(
    U: np.ndarray, V: np.ndarray, TH: np.ndarray
) -> np.ndarray:
    """Compute Ertel PV from ERA5 state (Wu conventions).

    Args:
        U:  Zonal wind, shape ``(10, 51, 87)`` [m/s].
        V:  Meridional wind, shape ``(10, 51, 87)`` [m/s].
        TH: Potential temperature, shape ``(10, 51, 87)`` [K].

    Returns:
        PV, shape ``(8, 51, 87)`` (interior σ-levels only).
    """
    from wu_python.core.pv_calc import compute_ertel_pv_wu
    return compute_ertel_pv_wu(U, V, TH)


def get_wu_lat_lon() -> Tuple[np.ndarray, np.ndarray]:
    """Return 1-D lat/lon arrays for the Wu NH domain.

    Returns:
        ``(lat, lon)``: latitudes [°N], longitudes [°E].
    """
    from wu_python.core.grid import lats, lons
    return lats.copy(), lons.copy()
