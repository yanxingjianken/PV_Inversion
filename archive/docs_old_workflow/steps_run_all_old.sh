#!/bin/bash
# run_all.sh — Execute the full PPVI pipeline end-to-end
# Usage: bash run_all.sh
# Requires: micromamba env fourcastnetv2, CDS API key (~/.cdsapirc)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="micromamba run -n fourcastnetv2 python"

echo "============================================"
echo " PPVI Pipeline — Davis Fig 8 Replica"
echo " CA Blocking Event 2025-01-08 00Z"
echo "============================================"
echo ""

run_step() {
    local step=$1
    local script=$2
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " STEP $step: $script"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    cd "$ROOT/$step"
    $PY "$script" 2>&1 | tail -5
    echo "  ✓ Step $step complete"
}

run_step "01_download"         "download_era5.py"
run_step "02_grid"              "grid_and_sigma.py"
run_step "03_climatology"        "clim_11day_mean.py"
run_step "04_write_grid"        "write_grid_files.py"
run_step "05_wu_pass_ab"       "wu_pass_ab.py"
run_step "06_wu_pass_c"        "wu_pass_c.py"
run_step "07_wu_pass_d"        "wu_pass_d.py"
run_step "08_parse_outputs"    "parse_and_pv.py"
run_step "09_pv_advection"     "pv_advection.py"
run_step "10_fig8"             "fig8_replica.py"

echo ""
echo "============================================"
echo " ALL 10 STEPS COMPLETE"
echo "============================================"
echo "Output figures:"
ls -la "$ROOT/data/figs/"
echo ""
echo "Key diagnostics:"
$PY -c "
import xarray as xr
ds = xr.open_dataset('$ROOT/data/wu_out/pv_advection.nc')
qa = ds.Q_anom_250.values
print(f'  PV anomaly @250hPa: [{np.nanmin(qa):.2f}, {np.nanmax(qa):.2f}] PVU')
for ip in range(3):
    pv = ds.PVadv.isel(piece=ip).values
    print(f'  Piece {ip+1} PVadv: p99={np.nanpercentile(np.abs(pv),99):.1f} PVU/day')
"
echo ""
echo "View: $ROOT/data/figs/fig8_replica.png"
