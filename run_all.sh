#!/bin/bash
# run_all.sh — Full PPVI pipeline (Wu/Davis piecewise PV inversion)
# Usage: bash run_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
STEPS="$ROOT/steps"
PY="micromamba run -n fourcastnetv2 python"

echo "============================================"
echo " PPVI Pipeline"
echo " Event: $(grep EVENT_DATE "$ROOT/config.py" | head -1)"
echo " Clim window: $(grep CLIM_WINDOW_DAYS "$ROOT/config.py" | head -1)"
echo "============================================"

run_step() {
    local name=$1 script=$2
    echo ""; echo "━━━ STEP $name ━━━"
    cd "$STEPS/$name"
    $PY "$script" 2>&1 | tail -3
    echo "  ✓ $name complete"
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
echo " PIPELINE COMPLETE"
echo " View: $ROOT/data/figs/fig8_replica.png"
echo "============================================"
