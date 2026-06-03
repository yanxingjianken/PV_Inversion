# wu_python/core/io.py — Wu .grid file reader and NetCDF writer
"""I/O utilities for the Wu PPVI pipeline.

read_wu_grid()  — Read a Wu-format .grid file → (H, TH, U, V) arrays
write_nc()      — Write results to NetCDF with CF-compliant metadata

IMPORTANT: The .grid file stores raw temperature T (Kelvin), NOT potential
temperature θ. The Fortran pvpialln_94UV.f converts T → θ internally via:
    TH(I,J,K) = TH(I,J,K) * CP / PI(K)
We apply the same conversion in read_wu_grid() so the returned TH matches
what the Fortran code uses internally. This ensures bit-identical PV,
STB, and all derivative computations.
"""

import numpy as np
import xarray as xr
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def read_wu_grid(filepath: str | Path) -> tuple[np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray]:
    """Read a Wu .grid file (4-field stacked: H, θ, U, V).

    Format: (10F8.1) fixed-width ASCII.
    Header: 10 values (domain bounds, grid spacing, NX, NY, 0, 0).
    Data:   (4*NW, NY, NX) stacked as H[NW,NY,NX] + T[NW,NY,NX]
            + U[NW,NY,NX] + V[NW,NY,NX].

    The .grid file stores temperature T (K). Fortran converts T → θ
    via TH*CP/PI(K). We replicate that conversion here so callers
    receive θ matching Fortran's internal state.

    Args:
        filepath: Path to .grid file.

    Returns:
        H  — geopotential height [m]  (NW, NY, NX)
        TH — potential temperature [K] (NW, NY, NX), Fortran-converted
        U  — zonal wind [m/s]          (NW, NY, NX)
        V  — meridional wind [m/s]     (NW, NY, NX)
    """
    from wu_python.core.nondim import CP, PI_VALS

    data = []
    with open(filepath) as f:
        for line in f:
            for tok in line.split():
                data.append(float(tok))

    hdr = np.array(data[:10])
    vals = np.array(data[10:])

    NX_g = int(hdr[6])
    NY_g = int(hdr[7])
    NW = 10  # Wu standard

    expected = 4 * NW * NY_g * NX_g
    if len(vals) != expected:
        raise ValueError(
            f"Grid data size mismatch: got {len(vals)}, expected {expected} "
            f"(4×{NW}×{NY_g}×{NX_g})"
        )

    # Reshape: (4*NW, NY, NX)
    arr = vals.reshape(4 * NW, NY_g, NX_g)

    # Split into 4 fields, each (NW, NY, NX)
    H  = arr[0*NW:1*NW]
    TH = arr[1*NW:2*NW].copy()  # T (raw temperature from .grid file)
    U  = arr[2*NW:3*NW]
    V  = arr[3*NW:4*NW]

    # Convert T → θ matching Fortran pvpialln_94UV.f:
    #   TH(I,J,K) = TH(I,J,K) * CP / PI(K)
    for k in range(NW):
        TH[k] *= CP / PI_VALS[k]

    return H, TH, U, V


def write_nc(ds: xr.Dataset, filepath: str | Path) -> None:
    """Write xarray Dataset to NetCDF, overwriting if exists."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        filepath.unlink()
    ds.to_netcdf(filepath)
