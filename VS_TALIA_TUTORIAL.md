# Our PPVI vs. the Talia tutorial — every algorithmic difference

Reference: `archive/talia_tutorial/inv3d/{pvpialln_94UV.f, qinvert21_94.f, qinvertp21_94.f}`
plus their input decks `*_in`.
Ours: `pvtend/src/pvtend/ppvi/{fortran/wuppvi.f, solver.py, winds.py}` +
`shared_steps/{loaders.py, ppvi_pipeline.py}`.

Method: both sides were stripped of comments, blanks and continuation formatting,
upper-cased and whitespace-collapsed, then diffed subroutine-by-subroutine.

---

## 0. Headline

**The numerics are the same code.** After normalisation, every SOR update
statement, every finite-difference coefficient and every physical constant in
`BALNC` and `BALP` is byte-identical to the tutorial:

| statement | identical? |
|---|---|
| `A(I,1..5)` horizontal Laplacian coefficients (incl. `SIGM²·APM/AP` metric terms) | yes |
| `AVO` absolute vorticity from the base-state ψ | yes |
| `STB` static stability from the base-state φ | yes |
| `RHS = QP + FR·((R1BS·R1PH + R1BH·R1PS)/AP² + SIG²·(R2BS·R2PH + R2BH·R2PS))` | yes |
| `SRHS`, `HRHS`, `ASI`, `BSI`, `APHI` | yes |
| `PI(K) = CP·PR(K)**KAP` (Exner vertical coordinate) | yes |
| `CP=1004.5`, `KAP=2/7`, `AA=2e7/π`, `GG=9.81`, `f=1.458e-4·sin φ` | yes |

Excluding I/O, declarations and DO-label renumbering, the residual diff is
**21 lines in `BALNC`** and **135 in `BALP`** — and *none* of them is a physics
statement. They are catalogued in §1.

Everything that actually differs scientifically is in §2.

---

## 1. Port changes that carry no physics

| # | Tutorial | Ours | Why |
|---|---|---|---|
| 1 | 3 `PROGRAM`s, 3 executables, files on disk between them | 5 `SUBROUTINE`s in one `wuppvi.f`: `PVPIALLN_CORE`, `QINVERT_CORE`+`BALNC`, `QINVERTP_CORE`+`BALP` | f2py; no intermediate files |
| 2 | `PARAMETER (NX=45)`, `PARAMETER (NY=21)`; grid dims also read as `HDR(7)`, `HDR(8)` | `NY`, `NX` are dummy arguments; `NYI`/`NXI` inside | one build serves every domain size |
| 3 | `READ(5,*)` / `PRINT*` / `WRITE(6,…)` / `FORMAT` throughout | all removed; parameters are arguments, fields are returns | non-interactive |
| 4 | labels `251`, `252`, `204`, `270`… | `2511`, `2521`, `2204`, `2270`… | collision-free after merging three files |
| 5 | `TH(I,J,K) = TH(I,J,K)*CP/PI(K)` (in-place on the θ array) | `TH(I,J,K) = TT(I,J,K)*CP/PI(K)` | `T` is a separate input; no aliasing |
| 6 | `IMAP=2` map-factor (x/y) coordinate path | dropped, `IMAP=1` (lat/lon) only | we never use a projected grid |
| 7 | `IBC=2` — read the first guess from a file | dropped; `IBC=0` (homogeneous) and `IBC=1` (total-perturbation BC) kept, we run `IBC=0` | no file plumbing |
| 8 | `DO 1000 IHALFDAY` — pass A/B loops over N half-day `.grid` files and accumulates the time mean | dropped | the mean state is built in Python (§2.4) |
| 9 | `DO 210 IH=1,NOUT` — one run inverts several pieces, reading a new `QLV` list per piece interactively | dropped | Python calls `qinvertp_core` once per piece |
| 10 | `IVNEG` negative-vorticity counter; every-5-iteration `DHMAX` residual report | dropped | diagnostics only |
| 11 | relied on `-fno-automatic` (static locals, zero-initialised) | compiled `-std=legacy -O2 -frecursive` (stack locals, **not** zero-initialised); `SISUM`/`HTSUM` explicitly zeroed, ~30 more locals explicitly declared | `-frecursive` is needed for process-fork safety |

> **⚠ #11 is only half-done and it is a live bug.** Only ~7 of ~35 `BALP` local
> arrays are explicitly zeroed. At `NL=9` the uninitialised remainder happens to
> be harmless; at `NL=20` pass D returns ~95 % NaN. This is the reason
> `_WU_PLEVS` is pinned to 9 levels. Fixing it means zeroing the other ~28
> arrays, not changing any physics.

---

## 2. Genuine algorithmic differences

### 2.1 Vertical grid — we dropped 150 hPa

```
tutorial  PR = 1.0, .85, .7, .5, .4, .3, .25, .2, .15, .1     NL = 10
ours      PR = 1.0, .85, .7, .5, .4, .3, .25, .2,      .1     NL =  9
```

Our level set is *exactly* the tutorial's with **150 hPa removed**. Consequence:
the topmost interval is 200→100 hPa, a 100-hPa jump in an upper troposphere that
is otherwise resolved at 50 hPa, so the Exner-space vertical second derivative at
K=8 (200 hPa) sits on a strongly asymmetric stencil. Restoring 150 hPa is a
one-line change *if* #11 is fixed first — going to `NL=10` re-enters the
uninitialised-locals regime.

### 2.2 Piece partition — 500 hPa changed sides

The tutorial's canonical deck (`qinvertp21_94_in`, `NOUT=3`):

```
1,1                     piece 1 = {1}         surface  (bottom boundary θ)
2,2,3                   piece 2 = {2,3}       850, 700
7,4,5,6,7,8,9,10        piece 3 = {4…10}      500 … 100  (incl. top boundary θ)
```

Ours (`shared_steps/ppvi_pipeline.py`):

```python
PIECES = {"surface": [1], "lower": [2, 3, 4],
          "upper_p": [5, 6, 7, 8, 9], "upper_e": [5, 6, 7, 8, 9]}
```

So **500 hPa belongs to `lower` for us and to `upper` for the tutorial.** Our
"upper" anomaly is 400–100 hPa; theirs is 500–100 hPa. Both put the top boundary
θ in the upper piece and the bottom boundary θ in the surface piece — that part
agrees.

### 2.3 The planetary/eddy split has no tutorial analogue

The tutorial can only partition **by level** — a piece *is* a list of `QLV`
indices. We added a **horizontal/spectral** partition of the *same* levels
through the new `qp_anoms` / `th_anoms` arguments:

```
QPIN_piece  = QBIN  + qp_anoms[name]      (instead of  Q_event − Q_mean)
THPIN_piece = THBIN + th_anoms[name]
```

an **additive PV-anomaly override**, not a multiplicative mask. This is legitimate
because pass D is linear in the PV source: arrays that sum to the full anomaly
give pieces that sum exactly to the unmasked result. A multiplicative weight
would need `filter(q')/q'`, which is unbounded where `q'` is small.

`th_anoms` exists because `qp_anoms` cannot reach the boundaries: K=1 and K=NL
are driven by `THPIN`, not `QPIN`, so masking the PV array leaves the boundary-θ
source untouched.

### 2.4 Mean state

| | tutorial | ours |
|---|---|---|
| where | inside `pvpialln`, `DO 1000 IHALFDAY` | Python, `_mean_state()` |
| sampling | the half-day `.grid` files you list — their deck has 20 = 10 days at 00/12 Z | daily over ±15 d = **31 samples**; the CESM archive climatology is 6-hourly over 300 member-years |
| domain | the same limited-area box | the whole NH band, cropped after |

The *definition* of the anomaly (instantaneous − time mean) is identical.

### 2.5 Domain and resolution

| | tutorial | ours |
|---|---|---|
| grid | 45 × 21 at ~2.5° | ±90° lon × 10.5–85.5 °N at ~1° (CESM: 1.25° × 0.94°) |
| extent | 112.5° lon × 50° lat | 180° lon × 75° lat |
| points/level | 945 | ~14 500 |

≈ 15× more points per level, ~28× including the ±90° padding. The wide box is
deliberate: the lateral BC has to sit ≥ 3 Rossby radii (L_R ≈ 837 km at 55 °N)
from the object.

### 2.6 Grid isotropy — why the pass-D header bug was invisible to them

The tutorial's 2.5° × 2.5° grid has `SIGM = dlon/dlat = 1` **exactly**, so
swapping `ZHDR(5)` and `ZHDR(6)` is a no-op there. All three tutorial programs use
the same convention (`SIGM=ZHDR(5)/ZHDR(6)`, `LL=AA·ZHDR(5)`,
rows at `ZHDR(3)−(I−1)·ZHDR(6)`), i.e. **the bug we just fixed was introduced by
our Python wrapper, not inherited**: passes A/B/C got the swapped `zhdr10`, pass D
got the raw `zhdr8`. It bit only the anisotropic grid:

| dataset | dlon/dlat | ⟨‖u_ppvi‖⟩/⟨‖u_obs‖⟩ before → after |
|---|---|---|
| ERA5 (1° × 1°) | 1.0 | unchanged (bit-identical) |
| JRA-3Q (resampled) | 1.0009 | 0.1 % change |
| CESM f09 | **1.33** | **1.66 → 1.01** |

### 2.7 Relaxation parameters — we are much more conservative

| | tutorial deck | ours |
|---|---|---|
| pass A/B | `IMAX=300`, `OMEGS=1.75`, `THRS=5e4` (hardcoded) | `imax=5000`, `omegs=1.75`, `thrs=5e4` |
| pass C | `MAXX/MAXXT=200`, `OMEGS=1.85`, `OMEGH=1.75`, `PART=0.5`, `THRSH=0.1` | `max_iter/max_outer=10000`, `omegs=1.4`, `omegah=1.4`, `part=0.05`, `thrsh=0.01` |
| pass D | `OMEGS=1.85`, `OMEGH=1.75`, `PART=0.8`, `THRS=0.1`, `BET=1`, `FR=1` | `omegs=1.85`, `omegah=1.4`, `part=0.05`, `thrsh=0.1` |

The under-relaxation is 10–16× heavier (`PART` 0.05 vs 0.5–0.8) and the iteration
cap 16–50× higher. SOR sweeps to converge scale with the domain's linear size, and
ours is ~28× the area; the aggressive `PART=0.8` that is stable on 45 × 21
diverges on ours.

### 2.8 We validate; the tutorial stops at ψ

The tutorial writes `(ψ′, φ′)` to `pert.out` and ends. We add:

- `psi_to_winds()` — Wu centred differences on ψ → `(u_rot, v_rot)` per piece;
- a closure check that Σ pieces = the unmasked inversion;
- a comparison against the **Helmholtz rotational wind**, computed with spherical
  harmonics **globally** and cropped afterwards (this is why `loaders.py` carries
  `U_glob`/`V_glob`: `sh_ops.is_nh_grid` only recognises an equator-to-pole
  hemisphere, so a 10.5–85.5 °N band would be treated as spanning 180°, scaling
  every meridional derivative by ~180/74).

### 2.9 Input state preparation

| | tutorial | ours |
|---|---|---|
| source | a `.grid` file of H, θ, U, V on the 10 levels, built externally | hybrid → pressure in Python: `p = hyam·P0 + hybm·ps` (CAM, top→surface) or `p = a + b·ps` (JRA, a in Pa, surface→top); ERA5 already isobaric |
| interpolation | none in the Fortran | `interp_monotonic` in log-p, with `np.maximum.accumulate` monotonicity enforcement |
| below ground | no treatment | `fill_below_ground()` before pass A |
| resolution | as supplied | ERA5/JRA strided to ~1° (`TARGET_DEG`); JRA's Gaussian band linearly resampled to uniform latitudes, because the Wu header carries a single `dlat` |

---

## 3. Summary

Nothing in the balance equations, the discretisation or the constants differs. The
scientific differences are five:

1. **150 hPa dropped** (NL 10 → 9) — forced by the `-frecursive` zero-init gap, not by choice.
2. **500 hPa moved from the upper piece to the lower piece.**
3. **A spectral planetary/eddy split** added on top of the level-wise partition, via additive PV/θ anomaly overrides.
4. **A much larger, finer, anisotropic domain**, with correspondingly conservative relaxation — and the anisotropy is what exposed the pass-D header bug.
5. **A wind-space validation stage** (Helmholtz closure) that the tutorial does not have.
