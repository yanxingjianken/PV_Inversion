"""Numba-parallel H solver kernel — extracted from balance.py for JIT compilation."""
import numpy as np
from numba import njit, prange

# Constants needed by the kernel (will be passed as arguments or compiled-in)
# These are small; passing as arguments is cleaner.

@njit(parallel=True, fastmath=True)
def _h_solve_column_parallel(
    H, ASI_all, RH_all, THA_nd, omegh,
    A_arr, BB_arr, BH_arr, BL_arr, DPI2_arr,
    ny, nx, nw
):
    """Parallel H solve: each (i,j) column solved independently via Thomas.
    
    Uses Jacobi-style horizontal coupling (all columns use old H for 
    horizontal Laplacian), enabling full column parallelism via prange.
    Returns max |ΔH| for convergence check.
    """
    n_int = nw - 2
    max_dh = 0.0
    
    # Parallelize over j (outer), serial i (inner) — each thread handles a j-slice
    for j in prange(1, nx - 1):
        # Per-thread tridiagonal work arrays
        a = np.zeros(n_int)
        b = np.zeros(n_int)
        c = np.zeros(n_int)
        d = np.zeros(n_int)
        
        for i in range(1, ny - 1):
            Ai2 = A_arr[i, 2]
            
            # Build tridiagonal system
            for k in range(1, nw - 1):
                ki = k - 1
                asi = ASI_all[k, i, j]
                
                # Horizontal Laplacian (Jacobi: use old H values from all neighbors)
                lap_h = (A_arr[i, 0] * H[k, i - 1, j] +
                         A_arr[i, 1] * H[k, i, j - 1] +
                         A_arr[i, 3] * H[k, i, j + 1] +
                         A_arr[i, 4] * H[k, i + 1, j])
                
                if k == 1:  # lower boundary
                    a[ki] = 0.0
                    b[ki] = Ai2 + asi * (BB_arr[k] + BL_arr[k])
                    c[ki] = asi * BH_arr[k]
                    d[ki] = (RH_all[k, i, j] - lap_h
                             - asi * THA_nd[i, j, 0] / DPI2_arr[k])
                elif k == nw - 2:  # upper boundary
                    a[ki] = asi * BL_arr[k]
                    b[ki] = Ai2 + asi * (BB_arr[k] + BH_arr[k])
                    c[ki] = 0.0
                    d[ki] = (RH_all[k, i, j] - lap_h
                             + asi * THA_nd[i, j, 1] / DPI2_arr[k])
                else:  # interior
                    a[ki] = asi * BL_arr[k]
                    b[ki] = Ai2 + asi * BB_arr[k]
                    c[ki] = asi * BH_arr[k]
                    d[ki] = RH_all[k, i, j] - lap_h
            
            # Thomas algorithm: forward elimination
            for k in range(1, n_int):
                w = a[k] / b[k - 1]
                b[k] -= w * c[k - 1]
                d[k] -= w * d[k - 1]
            
            # Back substitution
            d[n_int - 1] /= b[n_int - 1]
            for k in range(n_int - 2, -1, -1):
                d[k] = (d[k] - c[k] * d[k + 1]) / b[k]
            
            # Update H in-place
            for k in range(1, nw - 1):
                ki = k - 1
                H_old = H[k, i, j]
                H_new = H_old + (d[ki] - H_old)
                H[k, i, j] = H_new
                dh = abs(H_new - H_old)
                if dh > max_dh:
                    max_dh = dh
    
    return max_dh
