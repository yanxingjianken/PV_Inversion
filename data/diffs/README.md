# Fortran Source Comparison: wu/fortran/ vs archive/talia_tutorial/

Generated: 2026-06-01

## Summary

The `wu/fortran/` versions are the **production** Fortran sources customized for
the CA2025 blocking case (51×87 NH domain, 10 σ-levels). The `archive/talia_tutorial/`
versions are the **original tutorial** versions from C.-C. Wu / Chris Davis for a
coarser 21×45 test domain with different vertical levels.

**The algorithmic core is identical across all versions** — same SOR solver,
balance equations, Ertel PV formula, and non-dimensionalization.

## File-by-File Differences

### pvpialln_94UV.f — Ertel PV + Balanced ψ

| Aspect | talia_tutorial (inv3d) | wu/fortran (production) |
|--------|----------------------|------------------------|
| **Grid** | NY=21, NX=45 | NY=51, NX=87 |
| **σ-levels (PR)** | 1, .85, .7, .5, .4, .3, .25, .2, .15, .1 | 1, .925, .85, .7, .6, .5, .4, .3, .25, .2 |
| **Prompts** | `TYPE*` statements | `PRINT*` statements |
| **File I/O** | `STATUS='new'` | `STATUS='new'` |
| **inv3d_test_days** | STATUS='replace', debug prints, `HDRI` array | N/A |

### qinvert21_94.f — Total PV Inversion (Pass C)

| Aspect | talia_tutorial | wu/fortran |
|--------|---------------|------------|
| **σ-levels (PR)** | Coarse (as above) | Fine (as above) |
| **QMIN, MAX, MAXT** | Hardcoded (0.01, 5000, 500) | Read from stdin |
| **inv3d_test_days** | Same as inv3d | N/A |

### qinvertp21_94.f — Perturbation PV Inversion (Pass D)

| Aspect | talia_tutorial (inv3d) | wu/fortran |
|--------|----------------------|------------|
| **σ-levels** | Coarse | Fine |
| **QMIN** | 0.00001 | 0.01 |
| **MAX** | 400 | 5000 |
| **MAXT** | 400 | 500 |
| **inv3d_test_days** | QMIN=0.01, MAX=250, MAXT=100 | N/A |

### pvpialln_94UV_new2.f — Experimental Variant (talia_tutorial/inv3d_test_days only)

- **Status**: Abandoned / incomplete
- **Changes**: Added `character*7 str_format` for dynamic FORMAT generation (commented out),
  `HDRI(:)` array for grid dimension caching, wider output format `FORMAT(45F10.2)`
- **Does not exist in wu/fortran/**

## Diff Files

- `wu_vs_talia_pvpialln.diff` — Unified diff for pvpialln_94UV.f
- `wu_vs_talia_qinvert21.diff` — Unified diff for qinvert21_94.f
- `wu_vs_talia_qinvertp21.diff` — Unified diff for qinvertp21_94.f

## Conclusion

The `wu/fortran/` versions are the canonical versions for the PV inversion project.
All algorithmic logic is preserved from the original tutorial; only grid dimensions,
vertical levels, I/O formatting, and parameter tuning differ. The tutorial versions
serve as a reference for the original Wu/Davis implementation but use a coarser
domain not suitable for the CA2025 blocking case.
