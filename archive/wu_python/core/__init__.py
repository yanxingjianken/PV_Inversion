# wu_python/core — Wu algorithm implementation in pure Python
"""Core modules implementing the Wu 4-pass PPVI pipeline.

Modules:
    grid       — Grid metrics: Coriolis, cos(lat), map factors, FD coefficients
    nondim     — Wu non-dimensionalization scales (LL, FF, THO, BB/BH/BL, etc.)
    fd_ops     — Finite-difference operators (Laplacian, gradient, Jacobian, ∂/∂π)
    pv_calc    — Pass A/B: Ertel PV computation + balanced ψ from vorticity
    sor_solver — Red-black SOR Poisson solver (2D + 3D)
    balance    — Pass C: BALNC — total balanced PV inversion
    piecewise  — Pass D: BALP — piecewise perturbation inversion
    io         — Wu .grid file reader, NetCDF writer
"""
