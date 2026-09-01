"""Fixed-format text I/O for the Wu/Davis Fortran-77 PV-inversion code.

The Fortran reads ``.grid`` input files and writes ``.out`` files as
column-aligned text:

* ``.grid`` input: one header line of 8 numbers
  (SLAT WLON NLAT ELON DLON DLAT NX NY, read free-format), followed by four
  variable blocks in the order H, T, U, V.  Each block loops over the 9
  pressure levels bottom-up (1000 hPa first); each level is NY rows ordered
  north -> south; each row is NX values written with ``FORMAT(10F8.1)``
  (at most 10 values per line, each exactly 8 characters, 1 decimal).
* ``.out`` output: a free-format header (list-directed write, may span
  several lines) followed by blocks written with ``FORMAT(13F10.2)``
  (13 values per line, each exactly 10 characters, 2 decimals).

Because the Fortran re-reads these files with the same FORMAT statements,
column alignment is mandatory: values are concatenated with no separator and
adjacent negative values can touch.  All parsing here therefore slices
fixed-width columns instead of splitting on whitespace.
"""

import numpy as np

#: Pressure levels (hPa), bottom-up, in the exact order the Fortran expects.
PLEV = [1000, 850, 700, 500, 400, 300, 250, 200, 100]
NLEV = len(PLEV)

#: Grid-file value bounds implied by the 8-character F8.1 field.
_F81_MIN = -9999.9
_F81_MAX = 99999.9


def format_header(slat, wlon, nlat, elon, dlon, dlat, nx, ny):
    """Return the .grid header line (no newline).

    Free-format on the Fortran side; the fixed spacing (6 x F8.1 + 2 x I6)
    mirrors the layout of the reference tutorial files.
    """
    return (f"{slat:8.1f}{wlon:8.1f}{nlat:8.1f}{elon:8.1f}"
            f"{dlon:8.1f}{dlat:8.1f}{int(nx):6d}{int(ny):6d}")


def _write_block(fh, arr, ny, nx, per_line=10):
    """Write one (nlev, ny, nx) block in FORMAT(10F8.1) row records."""
    for lev in arr:
        for row in lev:
            for start in range(0, nx, per_line):
                chunk = row[start:start + per_line]
                fh.write("".join(f"{v:8.1f}" for v in chunk) + "\n")


def write_grid(path, header, H, T, U, V):
    """Write a .grid input file for the Fortran inversion code.

    Parameters
    ----------
    path : str
        Output file path.
    header : sequence of 8 numbers
        (SLAT, WLON, NLAT, ELON, DLON, DLAT, NX, NY).  SLAT/WLON are the
        south / west edges; WLON and ELON follow the -180..180 convention.
    H, T, U, V : array_like, shape (9, ny, nx)
        Geopotential height (m), temperature (K), zonal and meridional wind
        (m/s).  Level index runs bottom-up (1000 hPa first, PLEV order);
        rows run north -> south; columns run west -> east.
    """
    if len(header) != 8:
        raise ValueError(f"header must have 8 numbers, got {len(header)}")
    nx, ny = int(header[6]), int(header[7])
    blocks = [("H", H), ("T", T), ("U", U), ("V", V)]
    for name, arr in blocks:
        a = np.asarray(arr, dtype=np.float64)
        if a.shape != (NLEV, ny, nx):
            raise ValueError(
                f"{name}: expected shape ({NLEV}, {ny}, {nx}), got {a.shape}")
        if not np.isfinite(a).all():
            raise ValueError(f"{name}: non-finite values")
        # Every value must fit an 8-character F8.1 field.
        vmin, vmax = float(a.min()), float(a.max())
        if not (_F81_MIN < vmin and vmax < _F81_MAX):
            raise ValueError(
                f"{name}: values [{vmin}, {vmax}] out of F8.1 range "
                f"({_F81_MIN}, {_F81_MAX})")
    with open(path, "w") as fh:
        fh.write(format_header(*header) + "\n")
        for _, arr in blocks:
            _write_block(fh, np.asarray(arr, dtype=np.float64), ny, nx)


def read_fblock(fh, ny, nx, width=10, fmt_width=10):
    """Read one (ny, nx) field written by a Fortran fixed-format WRITE.

    Each grid row is its own Fortran record: nx values at ``width`` values
    per line, each value exactly ``fmt_width`` characters wide.  Values are
    recovered by slicing fixed-width columns (never ``split()``: adjacent
    negative values can touch).

    Use ``width=13, fmt_width=10`` for FORMAT(13F10.2) outputs and
    ``width=10, fmt_width=8`` for FORMAT(10F8.1) grid files.
    """
    out = np.empty((ny, nx), dtype=np.float64)
    for i in range(ny):
        vals = []
        remaining = nx
        while remaining > 0:
            line = fh.readline()
            if not line:
                raise EOFError(
                    f"unexpected EOF: row {i + 1}/{ny}, "
                    f"{remaining} values still expected")
            line = line.rstrip("\n")
            n_here = min(width, remaining)
            for j in range(n_here):
                field = line[j * fmt_width:(j + 1) * fmt_width]
                if not field.strip():
                    raise ValueError(
                        f"short record: row {i + 1}, value {len(vals) + 1}: "
                        f"{line!r}")
                vals.append(float(field))
            remaining -= n_here
        out[i, :] = vals
    return out


def _read_free_numbers(fh, n):
    """Consume whitespace-separated numbers from as many lines as needed."""
    vals = []
    while len(vals) < n:
        line = fh.readline()
        if not line:
            raise EOFError(f"EOF while reading free-format header "
                           f"({len(vals)}/{n} values)")
        vals.extend(float(tok) for tok in line.split())
    if len(vals) > n:
        raise ValueError(f"header: expected {n} numbers, got {len(vals)}")
    return vals


def read_out_blocks(path, block_specs, width=13, fmt_width=10,
                    n_header_values=8, return_header=False):
    """Read named blocks from a Fortran .out file.

    Parameters
    ----------
    path : str
        File written by the Fortran with FORMAT(13F10.2) records, preceded
        by a list-directed 8-number header (set ``n_header_values=0`` for
        headerless files).
    block_specs : list of (name, nlev, ny, nx)
        Blocks in file order.  ``nlev=1`` reads a single 2-D field; the
        returned array is always shaped (nlev, ny, nx).

    Returns
    -------
    dict name -> ndarray(nlev, ny, nx), plus the header list if
    ``return_header`` is true.
    """
    blocks = {}
    with open(path) as fh:
        header = (_read_free_numbers(fh, n_header_values)
                  if n_header_values else [])
        for name, nlev, ny, nx in block_specs:
            arr = np.empty((nlev, ny, nx), dtype=np.float64)
            for k in range(nlev):
                arr[k] = read_fblock(fh, ny, nx,
                                     width=width, fmt_width=fmt_width)
            blocks[name] = arr
    if return_header:
        return blocks, header
    return blocks
