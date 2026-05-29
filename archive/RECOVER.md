# Recovery Recipe — Wu/Davis Piecewise PV Inversion (PPVI)
## Target: Replicate Davis et al. (2022, J. Climate, JCLI-D-22-0674.1) Fig 8 for 2025-01-08 00Z California blocking event

**Date written**: 2026-05-28  
**Why this exists**: Working scripts in `case_jan2025_CA/scripts/` got contaminated after multiple edit cycles. This document is a self-contained recipe to rebuild the entire pipeline from scratch. **Do not reference the existing scripts** — rewrite from this spec.

---

## 0. Ground rules (NEVER VIOLATE)

| Rule | Why |
|------|-----|
| **NEVER touch** `/net/flood/data2/users/x_yan/pv_inversion/talia_tutorial/` or `/net/flood/data2/users/x_yan/pv_inversion/ppvi/` | Reference reproductions for the original tutorial; must remain pristine. |
| Use micromamba env `fourcastnetv2` for all Python | Already has xarray, cdsapi, scipy, cartopy. |
| Build Fortran with `gfortran -std=legacy -O2 -fno-automatic` | F77 fixed-form; `-fno-automatic` is required so SAVE'd arrays persist across BALP calls. |
| Put scratch files in `/net/flood/data2/users/x_yan/tmp/`, not `/tmp/` | NFS-backed, persists across reboots. |
| **Do NOT use** the Wu solver's Q output for plotting | It's in opaque solver-internal units (~600× PVU). Use ERA5 Ertel PV instead. The ψ output IS in physical m²/s and is fine. |

---

## 1. Hard-coded grid & vertical configuration

**Match all numbers exactly — Fortran arrays are statically dimensioned.**

| Symbol | Value | Notes |
|--------|-------|-------|
| `NX`   | 87    | longitudes |
| `NY`   | 51    | latitudes (north → south in array) |
| `NW`   | 10    | total pressure levels |
| `NW_PV`| 8     | interior PV levels (K=2..NL-1) |
| `dlat = dlon` | 1.5° | |
| Domain | `lat: 85.5N → 10.5N`, `lon: -169.5E → -40.5E` | CDS `area=[85.5, -169.5, 10.5, -40.5]` |
| Pressure levels (hPa) | `1000, 925, 850, 700, 600, 500, 400, 300, 250, 200` | |
| σ-coord (PR in Fortran) | `1.0, 0.925, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2` | uniform 0.05 spacing at top 3 levels |

**Critical**: top level is 200 hPa, **not** 100 hPa. Going to 100 hPa gives a 0.25→0.1 σ-jump that destabilizes SOR at K=NL-1=250 hPa. Verified: blows up regardless of MAX iterations.

---

## 2. Fortran sources — patches to apply BEFORE recompiling

Three executables in `/net/flood/data2/users/x_yan/pv_inversion/inv3d_ca2025/`:

| Source file | Output exec | Pass | Purpose |
|-------------|-------------|------|---------|
| `pvpialln_94UV.f`  | `pvpialln.exe`  | A+B | Mean state, Ertel PV, streamfunction |
| `qinvert21_94.f`   | `qinvert21.exe` | C  | Total balanced inversion (SOR) |
| `qinvertp21_94.f`  | `qinvertp21.exe`| D  | Piecewise perturbation inversion (BALP subroutine) |

### 2.1 Required patches to each source

**`pvpialln_94UV.f`** — line ~49:
```fortran
DATA PR/ 1.0, 0.925, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2/
```

**`qinvert21_94.f`** — line ~188:
```fortran
DATA PR/ 1., .925, .85, .7, .6, .5, .4, .3, .25, .2/
```

**`qinvertp21_94.f`** — **THREE edits**:
- Line ~82 (PR data):
  ```fortran
  DATA PR/ 1.0, 0.925, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2/
  ```
- Lines ~87–88 (SOR iteration caps — **default 250/100 is too small for 87×51 grid**):
  ```fortran
  PARAMETER (MAX=5000)
  PARAMETER (MAXT=500)
  ```
- PARAMETER blocks at line ~70-71 (main) AND ~317-318 (BALP) both need NX=87, NY=51:
  ```fortran
  PARAMETER (NX=87, NY=51, NL=10)
  ```

### 2.2 Build commands

```bash
cd /net/flood/data2/users/x_yan/pv_inversion/inv3d_ca2025
gfortran -std=legacy -O2 -fno-automatic -o pvpialln.exe   pvpialln_94UV.f
gfortran -std=legacy -O2 -fno-automatic -o qinvert21.exe  qinvert21_94.f
gfortran -std=legacy -O2 -fno-automatic -o qinvertp21.exe qinvertp21_94.f
```

Expected sizes ~35–45 KB each. Sanity-check no compilation errors (warnings about `goto` are normal for F77).

---

## 3. Wu file formats (read these carefully)

| File ext | Format | Contents |
|----------|--------|----------|
| `*.grid` (input) | `(10F8.1)` | header(8) + then for each level: NY*NX field as flat array |
| `meanq`, `event_q.out` | `(13F10.2)` | header(8) + θ_bottom(NY,NX) + θ_top(NY,NX) + NW_PV × Q(NY,NX) |
| `event_pert.out` | `(13F10.2)` | header(8) + per_piece × [NW × HP(NY,NX) + NW × SP(NY,NX)] |
| Header (8 floats) | `lat_S, lon_W, lat_N, lon_E, dlat, dlon, NX, NY` | `lat_S=10.5`, `lat_N=85.5`, `dlat=dlon=1.5`, `NX=87`, `NY=51` |

**Array ordering**: I=0 is northernmost row (85.5°N), I=NY−1 is southernmost (10.5°N). J=0 is westernmost (−169.5°E). Same convention used inside the Fortran solver — do NOT flip.

**Wu sentinels**:
- Q ≤ 0 cells are written as `9999.9` in Q files (PV interior solver requires Q > 0).
- These cells (typically 272/4437 ≈ 6% — low latitudes at upper levels) are masked, not used.
- HP/SP perturbations are physical real values.

**Unit conventions in outputs**:
- `SP` (perturbation ψ) stored as `actual_ψ / 1e5`, so multiply by 1e5 to get m²/s.
- `HP` (perturbation Φ) stored in metres.
- `Q` is in opaque ~600× PVU units (NOT PVU). **Don't use it for plotting.** Re-derive Ertel PV from ERA5 t,u,v in Python.

---

## 4. Pipeline — order of operations

```
01_download_era5.py     ← ERA5 event snapshot (t,u,v,z; 10 levels)
02_prep_clim.py         ← climatology from /net/flood/data2/users/x_yan/era/clim/
03_write_grid.py        ← write Wu .grid files (event + climatology)
04_run_wu.sh            ← four Wu Fortran passes
05_plot_diag.py         ← (optional) sanity-check diagnostics
06_read_outs.py         ← parse .out files → NetCDF + ERA5 PV
07_plot_fig8.py         ← 3-panel Davis Fig 8 replica
```

All scripts live under `/net/flood/data2/users/x_yan/pv_inversion/case_jan2025_CA/scripts/`. Data under `case_jan2025_CA/data/`. Wu I/O under `case_jan2025_CA/wu_inv/`. Plots under `case_jan2025_CA/figs/`.

---

## 5. `01_download_era5.py` — event snapshot via CDS API

```python
"""01 — Download ERA5 event snapshot (2025-01-08 00Z) for PPVI inversion."""
import cdsapi, os
OUT = "/net/flood/data2/users/x_yan/pv_inversion/case_jan2025_CA/data/era5"
os.makedirs(OUT, exist_ok=True)
LEVELS = ["1000","925","850","700","600","500","400","300","250","200"]
c = cdsapi.Client()
c.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        "product_type":   "reanalysis",
        "format":         "netcdf",
        "variable":       ["temperature","u_component_of_wind","v_component_of_wind","geopotential"],
        "pressure_level": LEVELS,
        "year":  "2025", "month": "01", "day": "08", "time": "00:00",
        "area":  [85.5, -169.5, 10.5, -40.5],   # N, W, S, E
        "grid":  ["1.5", "1.5"],
    },
    os.path.join(OUT, "era5_event_20250108_1p5deg.nc"),
)
```

Gotcha: **don't request supplemental clim from CDS in the same run** — large area × 30 years × hourly will hit "cost limits exceeded". Always do clim from the local archive (step 6 below).

---

## 6. `02_prep_clim.py` — climatology on the analysis grid

Source: `/net/flood/data2/users/x_yan/era/clim/era5_hourly_clim_1990-2020_jan_<var>.nc` for `var ∈ {t, u, v, z}`. Native levels: `{100, 200, 250, 300, 400, 500, 700, 850, 1000}`. **Need to insert 925 and 600 via log-p interpolation** to match the 10-level Wu grid.

```python
"""02 — Prep climatology (Jan 8 00Z mean) on 87×51 1.5° grid."""
import os, numpy as np, xarray as xr
CLIM_DIR = "/net/flood/data2/users/x_yan/era/clim"
OUT      = "/net/flood/data2/users/x_yan/pv_inversion/case_jan2025_CA/data/clim"
os.makedirs(OUT, exist_ok=True)
TARGET_LEVS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]
LAT_S, LAT_N = 10.5, 85.5
LON_W, LON_E = -169.5, -40.5

VARS = {"t": "t", "u": "u", "v": "v", "z": "z"}
for short, full in VARS.items():
    src = os.path.join(CLIM_DIR, f"era5_hourly_clim_1990-2020_jan_{short}.nc")
    d = xr.open_dataset(src)
    # select Jan 8 00Z
    d = d.sel(month=1, day=8, hour=0)
    # subset
    d = d.sel(latitude=slice(LAT_N, LAT_S), longitude=slice(LON_W, LON_E))
    # log-p interpolate to 925 and 600
    lev = d.pressure_level.values.astype(float)
    arr = d[full].values            # (lev, lat, lon), lev native ordering
    out = []
    for p in TARGET_LEVS:
        if p in lev:
            out.append(arr[list(lev).index(p)])
        else:
            # log-p interp between bracketing native levels
            below = lev[lev > p].min(); above = lev[lev < p].max()
            w = (np.log(p) - np.log(above)) / (np.log(below) - np.log(above))
            i_b = list(lev).index(below); i_a = list(lev).index(above)
            out.append(w * arr[i_b] + (1 - w) * arr[i_a])
    out = np.stack(out)             # (10, lat, lon)
    ds = xr.Dataset(
        {full: (("pressure_level","latitude","longitude"), out.astype(np.float32))},
        coords={
            "pressure_level": ("pressure_level", TARGET_LEVS),
            "latitude":       d.latitude,
            "longitude":      d.longitude,
        },
    )
    ds.to_netcdf(os.path.join(OUT, f"clim_jan08_00z_{short}.nc"))
    print(f"wrote {short}")
```

---

## 7. `03_write_grid.py` — produce Wu input `.grid` files

Wu needs separate files per variable per state (event vs clim). Format `(10F8.1)`.

Required output files in `wu_inv/`:
- `event_t.grid`, `event_u.grid`, `event_v.grid`, `event_z.grid`
- `mean_t.grid`, `mean_u.grid`, `mean_v.grid`, `mean_z.grid`

```python
"""03 — Write Wu .grid input files."""
import os, numpy as np, xarray as xr
CASE = "/net/flood/data2/users/x_yan/pv_inversion/case_jan2025_CA"
WU   = os.path.join(CASE, "wu_inv"); os.makedirs(WU, exist_ok=True)
DATA = os.path.join(CASE, "data")
TARGET_LEVS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]
NX, NY = 87, 51

def write_grid(arr3d, path):
    """arr3d: (NW=10, NY, NX), lats N→S already, lons W→E."""
    # Header: lat_S, lon_W, lat_N, lon_E, dlat, dlon, NX, NY
    hdr = [10.5, -169.5, 85.5, -40.5, 1.5, 1.5, float(NX), float(NY)]
    with open(path, "w") as f:
        # header in same (10F8.1) format
        vals = hdr + [0.]*2     # pad to 10
        f.write("".join(f"{v:8.1f}" for v in vals) + "\n")
        for k in range(arr3d.shape[0]):
            flat = arr3d[k].ravel()
            for i in range(0, len(flat), 10):
                f.write("".join(f"{v:8.1f}" for v in flat[i:i+10]) + "\n")

def load_xy(path, var):
    d = xr.open_dataset(path).squeeze()
    lev_dim = "pressure_level" if "pressure_level" in d.dims else "level"
    # sort so lat is N→S, lon W→E, lev follows TARGET_LEVS
    d = d.sortby("latitude", ascending=False).sortby("longitude")
    arr = d[var].transpose(lev_dim, "latitude", "longitude").values
    lev = d[lev_dim].values
    # reorder to TARGET_LEVS
    idx = [list(lev).index(p) for p in TARGET_LEVS]
    return arr[idx]

# Event
ev = os.path.join(DATA, "era5/era5_event_20250108_1p5deg.nc")
for short in ["t","u","v","z"]:
    write_grid(load_xy(ev, short), os.path.join(WU, f"event_{short}.grid"))

# Climatology
for short in ["t","u","v","z"]:
    path = os.path.join(DATA, f"clim/clim_jan08_00z_{short}.nc")
    write_grid(load_xy(path, short), os.path.join(WU, f"mean_{short}.grid"))

print("wrote 8 .grid files")
```

---

## 8. `04_run_wu.sh` — four Fortran passes

```bash
#!/bin/bash
set -e
cd /net/flood/data2/users/x_yan/pv_inversion/case_jan2025_CA/wu_inv
EXE=/net/flood/data2/users/x_yan/pv_inversion/inv3d_ca2025

# ── Pass A: compute mean Ertel PV + ψ from climatology ──
$EXE/pvpialln.exe << EOF
'mean_t.grid' 'mean_u.grid' 'mean_v.grid' 'mean_z.grid'
'meanq' 'meanh'
EOF

# ── Pass B: compute event Ertel PV + ψ from event snapshot ──
$EXE/pvpialln.exe << EOF
'event_t.grid' 'event_u.grid' 'event_v.grid' 'event_z.grid'
'event_q.out' 'event_h.out'
EOF

# ── Pass C: total balanced inversion (sanity check, not used in plot) ──
$EXE/qinvert21.exe << EOF
1.4, 1.4, 0.5, 0.01, 1., 1.
'event_q.out' 'event_h.out' 'event_bal.out'
1, 0
EOF

# ── Pass D: piecewise perturbation inversion (THE main pass) ──
$EXE/qinvertp21.exe << EOF
1.4, 1.4, 0.5, 0.01, 1., 1.
'meanq' 'meanh' 'event_q.out' 'event_bal.out' 'event_pert.out'
1, 0, 0
10,1,2,3,4,5,6,7,8,9,10
10,1,2,3,4,5,6,7,8,9,10
2, 1, 2
0
2, 3, 4
0
5, 5, 6, 7, 8, 9
0
EOF
```

### 8.1 Pass D stdin decoded (THIS IS CRITICAL)

```
OMEGS=1.4  OMEGH=1.4  PART=0.5  THRSH=0.01  TSCAL=1.  QSCAL=1.
files...
IMAP=1   INLIN=0   IQD=0          ← linear balance (MUST be INLIN=0)
HOUT  (which levels' HP to write):  10 levels listed
SIOUT (which levels' SP to write):  10 levels listed
piece 1: NIN=2 levels = {1, 2}        IBC=0
piece 2: NIN=2 levels = {3, 4}        IBC=0
piece 3: NIN=5 levels = {5, 6, 7, 8, 9}  IBC=0
```

| Symbol | Value | Why |
|--------|-------|-----|
| `OMEGS`, `OMEGH` | 1.4 | SOR relaxation; >1.5 oscillates, <1.2 too slow |
| `PART` | 0.5 | Under-relaxation between PSI and PHI updates |
| `THRSH` | 0.01 | Convergence threshold (1% of max residual) |
| `INLIN` | **0 (linear balance)** | **`INLIN=1` (nonlinear) blows up**: PSI denom `BSI·AC(3)−2·ASI·(SLL+SPP)` → 0 near 250-hPa jet → division explodes |
| `IBC`   | 0 | Homogeneous Dirichlet BC on side walls |

Pieces correspond to layer ranges (1-indexed Fortran K):
- piece 1 = K∈{1,2} = 1000–925 hPa = "lower troposphere"
- piece 2 = K∈{3,4} = 850–700 hPa = "middle troposphere"
- piece 3 = K∈{5..9} = 600–250 hPa = "upper troposphere"

**Expected**: each piece reports `TOTAL CONVERGENCE` within ≤25 outer iterations. If not, raise OMEGS in steps of 0.05, or revisit MAX/MAXT.

---

## 9. `06_read_outs.py` — parse Wu output → NetCDF (+ ERA5 PV)

Two NetCDFs to write into `data/wu_out/`:
- `piecewise_psi.nc`: HP, SP, induced U, induced V, dims (piece, plev, lat, lon)
- `pv_advection.nc`:  PVadv, Q_event_250, Q_clim_250, Q_anom_250, U_induced_250, V_induced_250

### 9.1 Read raw ASCII (whitespace-tokenized — safer than fixed-width)

```python
def read_wu_ascii(path):
    data = []
    with open(path) as f:
        for line in f:
            for tok in line.split():
                data.append(float(tok))
    return np.array(data[:8]), np.array(data[8:])
```

### 9.2 Wind derivation from ψ

```python
R_EARTH = 2.0e7 / np.pi          # ≈ 6.366e6 m  (matches Wu's AA constant)
DP = R_EARTH * np.radians(1.5)   # meridional grid spacing [m]
DL = R_EARTH * np.radians(1.5)   # zonal at equator
AP = np.cos(np.radians(lats))    # cos(lat) per row, lats N→S

psi = SP[piece] * 1.0e5          # restore actual ψ [m²/s]
# I increases southward → MATLAB convention:
U = np.zeros_like(psi)
U[:, 1:-1, :] = (psi[:, 2:, :] - psi[:, :-2, :]) / (2*DP)
V = np.zeros_like(psi)
for i in range(1, NY-1):
    V[:, i, 1:-1] = (psi[:, i, 2:] - psi[:, i, :-2]) / (2*DL*AP[i])
```

### 9.3 **CRITICAL**: derive Ertel PV at 250 hPa from ERA5, NOT from Wu Q

Wu's Q is in opaque solver-internal units (factor ~600× PVU, not 100× as I once assumed). Compute Ertel PV in Python directly:

```python
def ertel_pv_250(t3d, u3d, v3d, plev_Pa, lat, lon):
    """Returns PV at 250 hPa in PVU. Arrays ordered (lev, lat, lon)."""
    P0, Rd, Cp = 1.0e5, 287.0, 1004.0
    theta = t3d * (P0 / plev_Pa[:, None, None]) ** (Rd / Cp)
    dtheta_dp = np.gradient(theta, plev_Pa, axis=0)
    R = 6.371e6
    dlat_rad = np.deg2rad(np.gradient(lat))[None, :, None]
    dlon_rad = np.deg2rad(np.gradient(lon))[None, None, :]
    coslat = np.cos(np.deg2rad(lat))[None, :, None]
    dudy = np.gradient(u3d, axis=1) / (R * dlat_rad)
    dvdx = np.gradient(v3d, axis=2) / (R * coslat * dlon_rad)
    zeta = dvdx - dudy
    f = 2*7.292e-5*np.sin(np.deg2rad(lat))[None, :, None]
    g = 9.81
    pv = -g * (zeta + f) * dtheta_dp * 1.0e6      # PVU
    k250 = int(np.argmin(np.abs(plev_Pa - 25000.0)))
    return pv[k250]
```

Compute for both event and climatology; `Q_anom_250 = Q_event_250 - Q_clim_250`. Expected range ±5–7 PVU for a blocking event.

### 9.4 PV advection by induced winds

```python
q_fill = np.where(np.isnan(q_anom), np.nanmean(q_anom), q_anom)
dqdx = np.zeros_like(q_fill); dqdy = np.zeros_like(q_fill)
dqdy[1:-1, :] = (q_fill[:-2, :] - q_fill[2:, :]) / (2*DP)   # I↑ = south↑
for i in range(1, NY-1):
    dqdx[i, 1:-1] = (q_fill[i, 2:] - q_fill[i, :-2]) / (2*DL*AP[i])

PVadv = -(u_induced_250 * dqdx + v_induced_250 * dqdy) * 86400.0   # PVU/day
# mask 1-cell halo
PVadv[0,:] = PVadv[-1,:] = PVadv[:,0] = PVadv[:,-1] = np.nan
```

Expected piece-1/2 PV_adv: ±10–30 PVU/day. Piece-3: ±100–250 PVU/day (inflated by solver upper-BC noise — see §11).

---

## 10. `07_plot_fig8.py` — three-panel replica

Three rows (lower / middle / upper piece), shared horizontal extent. Each panel shows:
- **Filled colour**: PV advection by induced winds (RdBu_r diverging)
- **Black contours**: total 250-hPa PV anomaly (solid for +, dashed for −)
- **Black quivers**: induced wind vectors at 250 hPa
- Reference arrow `20 m/s` in lower-right

### 10.1 Critical defaults (DO NOT hardcode tight ranges)

```python
# Auto-scale PV advection vmax per panel (p98 → nice grid)
nice = np.array([0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100])
panel_vmax = []
for ip in range(3):
    v = np.nanpercentile(np.abs(PVadv[ip]), 98)
    v = float(nice[np.argmin(np.abs(nice - v))])
    panel_vmax.append(v)
# typical result: [10, 7.5, 30] PVU/day

# Auto-scale q_anom contour CI from p95 (target ~6 contours/side)
p95 = np.nanpercentile(np.abs(q_anom), 95)
nice_ci = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
ci = float(nice_ci[np.argmin(np.abs(nice_ci - p95/6))])
cmax = float(np.ceil(p95 / ci) * ci)
POS_LEVS = np.arange(ci, cmax + ci/2, ci)
NEG_LEVS = -POS_LEVS[::-1]
# typical result: CI=0.5 PVU, levels ±[0.5..4.5]
```

### 10.2 Piece-3 noise suppression

```python
from scipy.ndimage import gaussian_filter
for ip in range(3):
    sigma = 1.5 if ip == 2 else 0.0
    if sigma > 0:
        U_ind[ip] = gaussian_filter(np.nan_to_num(U_ind[ip]), sigma=sigma)
        V_ind[ip] = gaussian_filter(np.nan_to_num(V_ind[ip]), sigma=sigma)
        PVadv[ip] = gaussian_filter(np.nan_to_num(PVadv[ip]), sigma=sigma)

# Quiver display cap so piece-3 outliers don't dominate scale
WIND_DISPLAY_CAP = 40.0   # m/s
speed = np.sqrt(u**2 + v**2)
scale = np.where(speed > WIND_DISPLAY_CAP, WIND_DISPLAY_CAP/np.maximum(speed,1e-9), 1.0)
u, v = u*scale, v*scale
```

### 10.3 Output

```
figs/fig8_replica.png   (dpi=300, ~6 MB)
figs/fig8_replica.pdf   (vector, ~400 KB)
```

---

## 11. Known issues / outstanding bugs

| Issue | Severity | Root cause | Workaround applied | Real fix needed |
|-------|----------|------------|--------------------|-----------------|
| Wu Q in opaque units (~600× PVU) | medium | Solver-internal scaling in `pvpialln.f` `COEF=1E2*1E6*9.81*KAP*(CP**3.5)/P0` is not a clean PVU conversion | Derive Ertel PV from ERA5 in Python | Audit `COEF` derivation in Wu source; or just don't use it |
| Piece 3 (upper) induced winds ±100/280 m/s | high | Rigid-lid SOR at K=NL=200 hPa leaks gridpoint noise into K=NL-1=250 hPa; Qpert std grows 46→576 from 925→250 hPa | Gaussian σ=1.5 smoothing + 40 m/s display cap in Python | Add Shapiro filter to `qinvertp21_94.f` at K ≥ NL−2, OR raise lid with non-uniform σ FD |
| `INLIN=1` (nonlinear balance) explodes | high | PSI denom `BSI·AC(3)−2·ASI·(SLL+SPP)` violates ellipticity near 250 hPa jet | Use `INLIN=0` (linear balance) — losing some accuracy in strong-flow regions | Either restrict INLIN=1 to lower troposphere, or add ellipticity guard in BALP |
| Default `MAX=250, MAXT=100` insufficient | low | 87×51 grid needs more SOR sweeps to converge | Bumped to `5000/500` in Fortran source | None (already fixed in Fortran) |

---

## 12. Validation checks (run after rebuild)

```python
# After 06_read_outs.py:
# (1) ERA5 Ertel PV @250 hPa should be sane
#     event: min ~−1, max ~9, p99 ~7.8 PVU
#     clim : min ~0,  max ~6, p99 ~5.5 PVU
#     anom : min ~−5.7, max ~7, p1/p99 ~−4.8/+5.4 PVU
#
# (2) Induced wind ranges (interior, no halo):
#     piece 1 (lower):  U ±8,   V ±10   m/s
#     piece 2 (middle): U ±9,   V ±15   m/s
#     piece 3 (upper):  U ±100, V ±280  m/s   ← noisy, will be smoothed/capped
#
# (3) PV advection (ERA5 ∇q × Wu winds):
#     piece 1: p1/p99 ±9–14   PVU/day
#     piece 2: p1/p99 ±6–10   PVU/day
#     piece 3: p1/p99 ±90–100 PVU/day
#
# (4) event_pert.out:  expect ZERO NaN, ZERO Inf
#     ASCII sanity: `awk 'NF>0{print NF; exit}' event_pert.out` should print 13
```

Visual check: panels (a) and (b) should show a clear ridge (positive PV anomaly contours) over the eastern Pacific around 50°N, 130°W with a downstream trough over central US — the canonical Pacific blocking dipole. Filled colours should show cyclonic (red) advection on the upstream flank of the ridge and anticyclonic (blue) downstream. Panel (c) will still be noisy until the Fortran-side fix in §11 is implemented.

---

## 13. Source-file checklist (recompile required if any patch changed)

```
inv3d_ca2025/pvpialln_94UV.f   ── PR(10)={1.0..0.2}
inv3d_ca2025/qinvert21_94.f    ── PR(10)={1..0.2}
inv3d_ca2025/qinvertp21_94.f   ── PR(10), MAX=5000, MAXT=500, PARAMETER(NX=87,NY=51,NL=10) in both main and BALP
```

After patching: rebuild all three with `gfortran -std=legacy -O2 -fno-automatic`.

---

## 14. One-shot rebuild (TL;DR)

```bash
# 1. Patch + compile Fortran (only if changed)
cd /net/flood/data2/users/x_yan/pv_inversion/inv3d_ca2025
gfortran -std=legacy -O2 -fno-automatic -o pvpialln.exe   pvpialln_94UV.f
gfortran -std=legacy -O2 -fno-automatic -o qinvert21.exe  qinvert21_94.f
gfortran -std=legacy -O2 -fno-automatic -o qinvertp21.exe qinvertp21_94.f

# 2. Rewrite scripts 01-07 from §§5–10 of this doc (do NOT reuse contaminated copies)
mkdir -p /net/flood/data2/users/x_yan/pv_inversion/case_jan2025_CA/scripts

# 3. Run pipeline
cd /net/flood/data2/users/x_yan/pv_inversion/case_jan2025_CA
micromamba run -n fourcastnetv2 python scripts/01_download_era5.py
micromamba run -n fourcastnetv2 python scripts/02_prep_clim.py
micromamba run -n fourcastnetv2 python scripts/03_write_grid.py
bash scripts/04_run_wu.sh 2>&1 | tee wu.log | grep -E "TOTAL|TOO"
micromamba run -n fourcastnetv2 python scripts/06_read_outs.py
micromamba run -n fourcastnetv2 python scripts/07_plot_fig8.py

# 4. View
ls -la figs/fig8_replica.{png,pdf}
```
