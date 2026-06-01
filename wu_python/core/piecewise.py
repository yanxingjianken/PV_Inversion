# wu_python/core/piecewise.py — BALP: Piecewise perturbation inversion (Pass D)
"""Piecewise PV perturbation inversion.

Ports the BALP subroutine from qinvertp21_94.f:
  - Takes mean state (Q_mean, PSI_mean, H_mean) as linearisation reference
  - Splits ΔQ = Q_event − Q_mean into NPIECES vertical bands
  - For each piece, inverts only that piece's Q anomaly → ψ′_piece
  - Pieces sum to total ψ anomaly

The 3 pieces (default) are run in parallel via ProcessPoolExecutor.

Reference: qinvertp21_94.f, subroutine BALP.
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from wu_python.config import N_WORKERS, PIECES, NPIECES  # noqa: E402
from wu_python.core.balance import balnc_total  # noqa: E402


def _invert_one_piece(args: tuple) -> tuple[int, np.ndarray, np.ndarray]:
    """Invert a single perturbation piece (worker function for parallel execution).

    Args:
        args: (ip, Q_piece, PSI_mean, H_mean, params)

    Returns:
        (ip, PSI_piece, H_piece)
    """
    ip, Q_piece, PSI_mean, H_mean, params = args
    PSI_piece, H_piece, converged, n_iter = balnc_total(
        Q_piece, H_mean, PSI_mean, **params
    )
    return ip, PSI_piece, H_piece


def balp_pieces(
    Q_event: np.ndarray,
    Q_mean: np.ndarray,
    PSI_mean: np.ndarray,
    H_mean: np.ndarray,
    pieces: list[list[int]] | None = None,
    n_workers: int = N_WORKERS,
    parallel: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise perturbation PV inversion (Pass D).

    Args:
        Q_event:   Total event PV (NW, NY, NX)
        Q_mean:    Climatological mean PV (NW, NY, NX)
        PSI_mean:  Mean streamfunction (NW, NY, NX)
        H_mean:    Mean geopotential height (NW, NY, NX)
        pieces:    List of level-index lists for each piece.
                   Default: [[0,1,2], [3,4,5], [6,7]] (low, mid, upper)
        n_workers: Number of parallel workers
        parallel:  Whether to use parallel execution

    Returns:
        PSI_pieces: (NPIECES, NW, NY, NX) — ψ′ per piece
        H_pieces:   (NPIECES, NW, NY, NX) — Φ′ per piece
    """
    if pieces is None:
        pieces = PIECES

    NW = Q_event.shape[0]
    n_pieces = len(pieces)
    PSI_pieces = np.zeros((n_pieces, NW, *Q_event.shape[1:]))
    H_pieces = np.zeros_like(PSI_pieces)

    # Compute total anomaly
    Q_anom = Q_event - Q_mean

    # Prepare tasks
    tasks = []
    for ip, levs in enumerate(pieces):
        Q_piece = np.zeros_like(Q_mean)
        for k in levs:
            if 0 <= k < NW:
                Q_piece[k] = Q_anom[k]
        tasks.append((ip, Q_piece, PSI_mean, H_mean,
                      {"omegs": 1.4, "omegh": 1.4, "part": 0.5,
                       "thrsh": 0.01, "maxt": 500, "max_iter": 5000,
                       "inlin": 0}))

    if parallel and n_workers > 1:
        with ProcessPoolExecutor(max_workers=min(n_workers, n_pieces)) as executor:
            results = list(executor.map(_invert_one_piece, tasks))
    else:
        results = [_invert_one_piece(t) for t in tasks]

    for ip, PSI_p, H_p in results:
        PSI_pieces[ip] = PSI_p
        H_pieces[ip] = H_p

    return PSI_pieces, H_pieces
