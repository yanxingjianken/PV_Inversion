"""Shared pytest fixtures for test_poisson benchmarks."""

from __future__ import annotations

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════════════
#  Grid fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def wu_lat() -> np.ndarray:
    """Wu NH-domain latitudes [°N], ascending."""
    from .config import DLAT, NY, LAT_S
    # Wu grid: lat increases southward in the Fortran,
    # but Python stores ascending (south→north).
    return np.linspace(LAT_S, LAT_S + (NY - 1) * DLAT, NY)


@pytest.fixture(scope="session")
def wu_lon() -> np.ndarray:
    """Wu NH-domain longitudes [°E]."""
    from .config import DLON, NX, LON_W
    return np.linspace(LON_W, LON_W + (NX - 1) * DLON, NX)


@pytest.fixture(scope="session")
def global_lat() -> np.ndarray:
    """Global latitudes [°], ascending -90 → 90."""
    return np.linspace(-90, 90, 181)


@pytest.fixture(scope="session")
def global_lon() -> np.ndarray:
    """Global longitudes [°], 0 → 360."""
    return np.linspace(0, 360, 360, endpoint=False)


# ═══════════════════════════════════════════════════════════════════════════
#  Idealised test data
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def sph_harm_l5m3() -> tuple:
    """Spherical harmonic Y_5^3 test case on 181×360 global grid."""
    from .idealized import sph_harm_test
    return sph_harm_test(l=5, m=3, nlat=181, nlon=360)


@pytest.fixture(scope="session")
def gaussian_wu(wu_lat, wu_lon):
    """Gaussian bump on the Wu NH grid."""
    from .idealized import gaussian_bump_test
    return gaussian_bump_test(wu_lat, wu_lon, lat0=50.0, lon0=-100.0, sigma_deg=2.0)


@pytest.fixture(scope="session")
def tanh_wu(wu_lat, wu_lon):
    """Tanh front on the Wu NH grid."""
    from .idealized import tanh_front_test
    return tanh_front_test(wu_lat, wu_lon, lat_front=55.0, width_deg=1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  ERA5 / real-data fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def era5_event() -> tuple:
    """ERA5 event-day state: (H, TH, U, V) each (10, 51, 87)."""
    from .data_loader import load_wu_event
    return load_wu_event()


@pytest.fixture(scope="session")
def era5_event_vorticity(era5_event) -> np.ndarray:
    """ERA5 relative vorticity at event time, shape (10, 51, 87)."""
    from .data_loader import compute_vorticity_era5
    H, TH, U, V = era5_event
    return compute_vorticity_era5(U, V)


@pytest.fixture(scope="session")
def era5_event_pv(era5_event) -> np.ndarray:
    """ERA5 Ertel PV at event time, shape (8, 51, 87)."""
    from .data_loader import compute_pv_era5
    H, TH, U, V = era5_event
    return compute_pv_era5(U, V, TH)
