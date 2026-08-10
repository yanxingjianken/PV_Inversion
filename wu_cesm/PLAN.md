# PLAN — 2026-08-06 · upper-level 3-D-object PPVI rebuild

Scope spans two trees:
- `/net/flood/data2/users/x_yan/pv_inversion/` — `wu/`, `wu_cesm/` (piece redefinition + figures)
- `/net/flood/data2/users/x_yan/pvtend/outputs/` — the per-event npz rebuild

Nothing is launched until the single-event test in `wu_cesm/single_peak_blocking_evt/` is reviewed.

---

## PART 1 — move 500 hPa from `upper` into `lower`, regenerate all figures

Wu levels are 1-indexed: `1=1000, 2=850, 3=700, 4=500, 5=400, 6=300, 7=250, 8=200, 9=100`.

| piece | now | after |
|---|---|---|
| surface | `{1}` — 1000 (θ) | unchanged |
| lower | `{2,3}` — 850–700 | **`{2,3,4}` — 850–500** |
| upper | `{4..9}` — 500–200 + 100 (θ) | **`{5..9}` — 400–200 + 100 (θ)** |

**Files to edit (two, and they must agree):**
- `wu/wu_config.yaml` → `pieces:` block. This is the AUTHORITATIVE copy; the step scripts read it.
- `config.py` → `PIECES` list (lines 66-70). Mirrors the YAML; `config.py:64` says so explicitly.

**Regeneration:** `bash run_all.sh`. Its Phase 0 already deletes every `*.png` under `shared_steps/`
and `wu/steps/`, so "清空所有图片然后重新生成" is exactly what the runner does. 14 PNGs are
regenerated:

```
steps/05_wu_pass_ab/{wu_mean_psi_500hpa, wu_mean_pv_3levels, wu_pv_anomaly_3levels}.png
steps/06_wu_pass_c/{pass_c_psi_all_levels, pass_c_psi_refinement}.png
steps/07_wu_pass_d/{pass_d_piece3_noise, pass_d_psi_3pieces_250hpa}.png
steps/08_parse_outputs/pv_comparison_250hPa.png
steps/10_fig8/fig8_replica.png
steps/11_xsection/{closure_check, closure_diagnostic, upper_psi_by_piece_250hpa,
                  xsection_pv_psi_4panel, xsection_vinduced_3panel}.png
```

Then `wu_cesm/closure_check.png` via `python wu_cesm/test_one_case.py` (which also rewrites
`closure_report.txt`).

**Watch for:** panel titles that hard-code the hPa range. `xsection_pv_psi_4panel.py:72-73` and
`10_fig8/_archive_pre9lev/*.py` read `piece_hpa[...]` from config, so they follow automatically —
but any string literal "500–200" would not. Will grep before running.

---

## PART 2 — rebuild the per-event npz: upper-level 3-D object, k≤4 split, per-level winds

### Current state of the four output folders

| folder | npz | size |
|---|---|---|
| `cesm_blocking` | **0** | empty |
| `cesm_blocking_wavg` | 18,501 | 45 G |
| `cesm_prp` | **0** | empty |
| `cesm_prp_wavg` | 79,083 | 194 G |

Two are already empty, so "合并为只有两个文件夹" = the new build writes `cesm_blocking/` and
`cesm_prp/`, and the `_wavg` pair is deleted **only after** the new build is verified.

### What changes vs. the current pipeline

| | now | after |
|---|---|---|
| object | 2-D mask from the wavg archive PV anomaly, broadcast to all 9 levels | **3-D connected** negative-PV object |
| object field | FULL spectrum | see Q2 |
| inverted PV | `M × (full 9-level perturbation)` | k≤4 part only, `pv_anom_p` |
| PPVI source | all pieces (`[1] / [2..8] / [9]` summed) | **upper piece only** |
| wind output | wavg-collapsed 2-D only | **per upper hPa level**, plus wavg |

### Pipeline per event

1. crop the ±90° × Wu-band window, 9 levels, event + climatology
2. build the 3-D connected negative-PV object on the upper levels → `M(x,y,p)`
3. `pv_anom_p` = its k≤4 part; `pv_anom_e = pv_anom − pv_anom_p`
4. PPVI with the **upper piece as the only source**, once for `pv_anom_p` and once for the total
5. `psi → u,v` at each upper level; `V_e = V_tot − V_p`
6. store per-level `u/v_block`, `u/v_eddy`, **plus** the wavg collapse of everything

### Cost — MEASURED 2026-08-06, and the ~25 h estimate was wrong

Still 2 pass-D inversions per event, but timed rather than assumed:

| `thrsh` | per inversion | per event | 97.6k events, `--jobs 48` |
|---|---|---|---|
| 1e-3 | 157 s | 314 s | **7.4 days** |
| 1e-2 | 73 s | 145 s | **3.4 days** |

The two tolerances give the SAME solution (psi to 0.1%, wind to 0.14%, corr 1.0000), so **1e-2 is
the production setting**. `1e-4` is broken outright — it returns a larger psi with an exactly zero
wind, i.e. psi collapsed to a constant.

The solver is NOT hitting `MAXIT`: wall time varies smoothly with log(1/thrsh), which a hard cap
could not produce. So the 10x `MAXIT` raise is free here, and the earlier `cesm_*_wavg` build was
not silently truncated at 2000 iterations.

---

## PART 3 — single-event test FIRST, in `wu_cesm/single_peak_blocking_evt/`

One BLOCK event at **peak**. Four numbered scripts, each with its figure.

| # | script | figure |
|---|---|---|
| 01 | `01_pv_anom_sections.py` | 2 panels: y–p section, and x–y at 250 hPa — `pv_anom` shaded with the `_p` object contoured on top |
| 02 | `02_psi_induced.py` | y–p of the 3-D ψ induced by `pv_anom_p` (left) and by `pv_anom_e` (right) |
| 03 | `03_winds_250.py` | x–y at 250 hPa: `u/v_rot_anom`, `u/v_block`, `u/v_eddy`, closure |
| 04 | `04_pv_closure.py` | same style as `wu_cesm/closure_check.png`, but checking **Wu PV(block) + Wu PV(eddy) vs the input Ertel `pv_anom`** |

Note on 04: the existing `closure_check.png` compares Wu `Q_event` against the input Ertel PV
(`corr=0.9578, NRMS=0.1058 @250`). The new one is a different check — that the two PIECES of the Wu
PV re-sum to the input anomaly — so both are worth keeping.

---

## DECISIONS (answered 2026-08-06)

| Q | answer |
|---|---|
| **Q1 object** | 3-D connectivity over the **upper levels only**. Threshold is **to be TUNED**, not fixed at 0.25 — it must work across all `onset/peak/decay` × `dh` × `prp/blocking`. Keep the Gaussian smoothing to a `[0,1]` weight. |
| **Q2 order** | **(b)** — filter FIRST, find the object on the k≤4 field: `pv_anom_p = M[Π_{k≤4}q'] · Π_{k≤4}q'`. Keeps `_p` confined to the object. |
| **Q3 pieces** | Wu `upper` **{4..9} → {5..9}**; level 4 (500 hPa) joins `lower`. Levels 1 (1000) and 9 (100) stay boundary-θ only, exactly as the Wu algorithm has it. |
| **Q4 wavg** | **unchanged** — still 300/250/200. |
| **Q5 folders** | New folders are literally `cesm_blocking/` and `cesm_prp/`. Delete the `_wavg` pair afterwards, but the new folders must carry **every** field the old ones had (`u_div_anom`, `Q`, `Q_lhr`, closure diagnostics, PV gradients, …) **plus** the new per-level winds. |
| **Q6 budget** | **Whole budget recomputed** from scratch. |
| **Q7 test case** | A **random BLOCK peak event at dh=0**. |

### Note on the level indexing in Q3
`WU = [1000, 850, 700, 500, 400, 300, 250, 200, 100]`, 1-indexed, so **500 hPa is level 4** and
400 hPa is level 5. The instruction said "make 5, as 500 hPa, to lower" but also gave the range
"4-9 → 5-9", which is unambiguous and is what moves 500 hPa out of `upper`. Implementing the RANGE:

```
lower = {2,3,4} = 850, 700, 500
upper = {5,6,7,8,9} = 400, 300, 250, 200 (interior PV) + 100 (top θ)
```

---

## FINAL SPEC (all questions answered 2026-08-06)

| item | decision |
|---|---|
| **pieces** | `lower = {2,3,4}` = 850/700/500 · `upper = {5..9}` = **400**/300/250/200 + 100 (θ). 400 hPa IS upper. |
| **dh scope** | **`dh=0` only.** Threshold tuned over `onset/peak/decay` × `dh=0` × `prp/block` = 6 combinations. |
| **threshold f** | **0.25** (user decision 2026-08-06). The three acceptance criteria only rule out f > 0.85 — they are satisfied for every f in 0.10-0.80, so they bound the threshold but cannot pick it. Object then occupies ~13% of the upper window, matching the existing pipeline's object size. |
| **object** | 3-D connected component, **6-connectivity**, over **400/300/250/200** (100 hPa excluded — see below), found on the **k≤4** field (order (b)), **hard 0/1, NO smoothing** (revised 2026-08-06). |
| **`pv_anom_p`** | `M[Π_{k≤4}q'] · Π_{k≤4}q'` |
| **100 hPa** | **NOT an object level** (revised 2026-08-06). It still contributes the top boundary θ to the PPVI, masked by `M` at **200 hPa**. |
| **per-level winds** | written at the four INTERIOR upper levels **400/300/250/200**. |
| **wavg** | unchanged, 300/250/200. |
| **budget** | whole budget recomputed. |
| **folders** | `cesm_blocking/`, `cesm_prp/`; every field the `_wavg` pair had must be reproduced; delete `_wavg` after verification. |
| **test case** | random BLOCK peak event, dh=0, in `single_peak_blocking_evt/`. |

### Threshold-tuning objective (final)

Sweep `f` and pick the value maximising the fraction of events where ALL of:

1. the object is **non-empty** and **contains the tracked centre at 250 hPa**;
2. the object is **3-D connected** (6-connectivity);
3. the object spans **≥ 2 pressure levels**.

No area cap — dropped at the user's instruction.

Validation set: `onset/peak/decay` × `dh=0` × `prp/block`, sampled across members/decades.

## SUPERSEDED — original question list (kept for provenance)

**Q1. The 3-D connected object**
- (a) connectivity over which levels — only the upper piece (400/300/250/200), or all 9?
- (b) threshold still `pv_anom < 0.25 × pv_anom[centre]`, seeded at the tracked block centre?
- (c) still Gaussian-smoothed to a `[0,1]` weight (σ=3 grid points), or now a hard 0/1 object?

**Q2. Order of "3-D object" and "k≤4" — these give different fields**
- (a) object on the FULL-spectrum anomaly, then filter: `pv_anom_p = Π_{k≤4}[M · q']`
- (b) filter first, then find the object on the k≤4 field:
  `pv_anom_p = M[Π_{k≤4}q'] · Π_{k≤4}q'`

  (a) has a wrinkle: a zonal-wavenumber filter applied to a spatially localised field spreads that
  object's signal back around the whole latitude circle, so `pv_anom_p` would no longer be confined
  to the object. (b) keeps it confined. Your sentence reads more like (a) to me, but I want it
  confirmed rather than guessed.

**Q3. PPVI source and output levels**
- source: `qlv` = the upper piece only (`{5..9}`), i.e. the Wu machinery's own piece selection?
- output: 400/300/250/200 — is **100 hPa** wanted too? It is the top boundary-θ piece and carries no
  interior PV, so there is no `u/v_block` "at" 100 hPa in the same sense.

**Q4. Does `wavg` change?**
Currently `wavg = 300/250/200` (`WAVG_LEV_IDX`). With `upper` now 400–200, should wavg become
**400/300/250/200**, or stay 300/250/200? This changes every wavg field in every npz.

**Q5. Folder names and the old data**
- new folders literally `cesm_blocking/` and `cesm_prp/`?
- delete the 239 GB `_wavg` pair after verification, or keep them?

**Q6. Which budget terms get recomputed?**
The npz currently holds the full budget (`lin_adv`, `baroclinic`, `div`, `vertical`, `Q_lhr`,
`dqdt`, closure diagnostics, PV gradients…). Only the `rot_nl` family and the winds depend on the
new split. Do you want:
- (a) only `rot_nl_{pp,pe,ep,ee}` + winds + `pv_anom_{p,e}` recomputed, everything else copied
  across unchanged, or
- (b) the whole budget recomputed from scratch (cleaner provenance, same cost since PPVI dominates)?



---

## REVISIONS — 2026-08-06, after the first single-event test

### R1 · Gaussian smoothing REMOVED

The object was Gaussian-smoothed (σ=3) to a `[0,1]` weight. On the test event this produced:

```
    400 hPa: binary     0 points     M>0.5    70 points     <-- object where there is none
    300 hPa: binary   226 points     M>0.5    73 points
    250 hPa: binary   305 points     M>0.5    83 points
    200 hPa: binary   389 points     M>0.5    90 points
```

The smoothing kernel bled the 300 hPa object downwards into 400 hPa, where the binary object is
empty. It was therefore not merely softening an edge — it was **extending the object's vertical
span**, which is one of the three acceptance criteria (`nlev >= 2`). Every "object spans N levels"
number reported before this revision is inflated. `M` is now the binary object as a float.

### R2 · 100 hPa EXCLUDED from the object

`pv9_15deg` has only **~47% valid PV at 100 hPa**, and the gaps are latitude-dependent, worst
exactly where blocks live:

```
    85.5N:  0.0% valid        48.0N: 69.4%
    78.0N:  5.8%              40.5N: 60.3%
    70.5N: 30.6%              25.5N: 71.9%
    63.0N: 50.4%              18.0N: 56.2%
    55.5N: 48.8%              10.5N: 38.8%
```

Verified in the raw archive before any climatology is subtracted (47.4% at 100 hPa vs **100.0%** at
250 hPa), so it is a property of the dataset, not of this pipeline. 1000 hPa is similarly sparse
(44.9%) but that is below-ground terrain and expected; 100 hPa is not.

Seeding a 3-D connected object on a half-missing field would make the object's top depend on where
the archive happens to have data. So the object is built on **400/300/250/200**, and the 100 hPa
top boundary θ — which is 100% valid, since θ needs no vertical derivative — is masked by `M` at
200 hPa.

*(An alternative was to mask 100 hPa by its own θ' rather than its PV', which is arguably more
faithful to the Wu formulation since level 9 contributes only θ. Not taken — the user chose the
simpler exclusion.)*

**Carry this forward:** any analysis that uses 100 hPa PV from `pv9_15deg` is working with a
half-empty field, biased against high latitudes.

### R3 · what figures 02-04 actually check

| # | naive reading | why it would be vacuous | what is checked instead |
|---|---|---|---|
| 03 | closure `V_p + V_e - V_tot` | identically 0 — `psi_e := psi_tot - psi_p`, psi→wind is linear | `(V_p + V_e) - V_rot_anom` against the **external** Helmholtz field. Not expected to vanish: the PPVI source is the upper piece only, so the residual **is** the lower+surface contribution. |
| 04 | `Q_p + Q_e` vs input Ertel `q'` | `Q_e := Q_tot - Q_p`, so it collapses to the existing `closure_check.png` | the **Ertel split vs the Wu split, piece by piece**. The budget differentiates Ertel `∇q'_p`; the winds come from inverting the Wu PV split. Same mask M, two different PV variables. If they disagree, `-V_p·∇q_e` is internally inconsistent and nothing downstream would reveal it. |

Both identities are still reported as numbers, so the arithmetic is on the record without a panel
of zeros standing in for a test.
