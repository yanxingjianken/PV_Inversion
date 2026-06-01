# wu_python/core/io.py — Wu .grid file reader and NetCDF writer
"""I/O utilities for the Wu PPVI pipeline.

read_wu_grid()  — Read a Wu-format .grid file → (H, TH, U, V) arrays
write_nc()      — Write results to NetCDF with CF-compliant metadata
"""

import numpy as np
import xarray as xr
from pathlib import Path


def read_wu_grid(filepath: str | Path) -> tuple[np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray]:
    """Read a Wu .grid file (4-field stacked: H, θ, U, V).

    Format: (10F8.1) fixed-width ASCII.
    Header: 10 values (domain bounds, grid spacing, NX, NY, 0, 0).
    Data:   (4*NW, NY, NX) stacked as H[NW,NY,NX] + TH[NW,NY,NX]
            + U[NW,NY,NX] + V[NW,NY,NX].

    Args:
        filepath: Path to .grid file.

    Returns:
        H  — geopotential height [m]  (NW, NY, NX)
        TH — potential temperature [K] (NW, NY, NX)
        U  — zonal wind [m/s]          (NW, NY, NX)
        V  — meridional wind [m/s]     (NW, NY, NX)
    """
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
    TH = arr[1*NW:2*NW]
    U  = arr[2*NW:3*NW]
    V  = arr[3*NW:4*NW]

    return H, TH, U, V


def write_nc(ds: xr.Dataset, filepath: str | Path) -> None:
    """Write xarray Dataset to NetCDF, overwriting if exists."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        filepath.unlink()
    ds.to_netcdf(filepath)
