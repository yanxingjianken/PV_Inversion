# wu_python — Pure-Python Wu PPVI Track (Finite Difference + Red-Black SOR)
"""Pure-Python reproduction of the Wu/Davis piecewise PV inversion pipeline.

Uses finite differences (NOT spherical harmonics) on the same 51×87 NH grid
as the Fortran reference, with red-black SOR parallelized via numba JIT and
ProcessPoolExecutor for pass/piece-level parallelism.

Default parallelism: 32 workers (configurable via N_WORKERS).
"""
