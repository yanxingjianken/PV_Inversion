#!/bin/bash
# run_all.sh — Full WuPPVI pipeline (download → wu step 11 xsection)
# Usage: bash run_all.sh
# Env:   micromamba run -n blocking python3
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SHARED="$ROOT/shared_steps"
WU="$ROOT/wu/steps"
PY="micromamba run -n blocking python3"

echo "============================================"
echo " WuPPVI Full Pipeline  (IBC=1)"
echo " Root:  $ROOT"
echo " Env:   blocking"
echo " Date:  $(date)"
echo "============================================"

# ── Phase 0: Clean ──
echo ""
echo "━━━ Phase 0: Clean intermediates ━━━"
echo "  Removing data/clim/*.nc ..."
rm -f "$ROOT/data/clim/mean_clim.nc" "$ROOT/data/clim/event.nc"
echo "  Removing data/wu_in/* (keep .grid) ..."
rm -f "$ROOT/data/wu_in/"*.out "$ROOT/data/wu_in/meanq" "$ROOT/data/wu_in/meanh" "$ROOT/data/wu_in/dummy"*
echo "  Removing data/wu_out/*.nc ..."
rm -f "$ROOT/data/wu_out/"*.nc
echo "  Removing data/wu_bin/*.exe ..."
rm -f "$ROOT/data/wu_bin/"*.exe "$ROOT/data/wu_bin/"*.o "$ROOT/data/wu_bin/"*.mod
echo "  Removing data/wu_in/*.grid ..."
rm -f "$ROOT/data/wu_in/"*.grid
echo "  Removing step plots ..."
find "$SHARED" "$WU" -name "*.png" -o -name "*.pdf" | xargs rm -f 2>/dev/null || true
echo "  Removing __pycache__ ..."
find "$ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Clean complete"

# ── Phase 1: Shared Steps ──
echo ""
echo "━━━ Phase 1: Shared Steps (ERA5 → .grid) ━━━"

echo "  Step 01 — Download + z500 plot ..."
cd "$SHARED/01_download"
$PY download_era5.py 2>&1 | tail -5
echo "    ✓ z500_evolution_5panel.png"

echo "  Step 02 — Grid & sigma viz ..."
cd "$SHARED/02_grid"
$PY grid_and_sigma.py 2>&1 | tail -3
echo "    ✓"

echo "  Step 03 — 30-year climatology → data/clim/ ..."
cd "$SHARED/03_climatology"
$PY clim_30yr_mean.py 2>&1 | tail -5
echo "    ✓ mean_clim.nc, event.nc"

echo "  Step 04 — Write Wu .grid files ..."
cd "$SHARED/04_write_grid"
$PY write_grid_files.py 2>&1 | tail -5
echo "    ✓ mean.grid, event.grid"

# ── Phase 2: Wu Fortran Solver ──
echo ""
echo "━━━ Phase 2: Wu Fortran Solver ━━━"

echo "  Step 05 — Compile + Pass A/B (mean PV, event PV) ..."
cd "$WU/05_wu_pass_ab"
$PY wu_pass_ab.py 2>&1 | tail -10
echo "    ✓ wu_mean_pv_3levels.png, wu_pv_anomaly_3levels.png, wu_mean_psi_500hpa.png"

echo "  Step 06 — Pass C (total balanced inversion) ..."
cd "$WU/06_wu_pass_c"
$PY wu_pass_c.py 2>&1 | tail -10
echo "    ✓ pass_c_psi_refinement.png, pass_c_psi_all_levels.png"

echo "  Step 07 — Pass D (3-piece parallel piecewise inversion) ..."
cd "$WU/07_wu_pass_d"
$PY wu_pass_d.py 2>&1 | tail -10
echo "    ✓ pass_d_psi_3pieces_250hpa.png, pass_d_piece3_noise.png"

# ── Phase 3: Post-process + Figures ──
echo ""
echo "━━━ Phase 3: Post-process + Figures ━━━"

echo "  Step 08 — Parse outputs + 30yr PV comparison ..."
cd "$WU/08_parse_outputs"
$PY parse_and_pv.py 2>&1 | tail -10
echo "    ✓ piecewise_psi.nc, pv_advection.nc, pv_comparison_250hPa.png"

echo "  Step 10 — Fig 8 replica ..."
cd "$WU/10_fig8"
$PY fig8_replica.py 2>&1 | tail -3
echo "    ✓ fig8_replica.png, fig8_replica.pdf"

# ── Phase 4: Step 11 X-Section Diagnostics (4 scripts in parallel) ──
echo ""
echo "━━━ Phase 4: Step 11 X-Section Diagnostics (parallel) ━━━"

cd "$WU/11_xsection"

echo "  Step 11a–11d — running 4 diagnostic scripts in parallel ..."
$PY closure_check.py 2>&1 | tail -5 > /tmp/_s11a.log 2>&1 &
_pid_11a=$!
$PY upper_psi_by_piece.py 2>&1 | tail -5 > /tmp/_s11b.log 2>&1 &
_pid_11b=$!
$PY xsection_vinduced.py 2>&1 | tail -5 > /tmp/_s11c.log 2>&1 &
_pid_11c=$!
$PY closure_diagnostic.py 2>&1 | tail -5 > /tmp/_s11d.log 2>&1 &
_pid_11d=$!

wait $_pid_11a $_pid_11b $_pid_11c $_pid_11d

echo "    11a closure_check: $(tail -1 /tmp/_s11a.log 2>/dev/null || echo done)"
echo "    11b upper_psi_by_piece: $(tail -1 /tmp/_s11b.log 2>/dev/null || echo done)"
echo "    11c xsection_vinduced: $(tail -1 /tmp/_s11c.log 2>/dev/null || echo done)"
echo "    11d closure_diagnostic: $(tail -1 /tmp/_s11d.log 2>/dev/null || echo done)"
rm -f /tmp/_s11{a,b,c,d}.log

# ── Done ──
echo ""
echo "============================================"
echo " PIPELINE COMPLETE"
echo " Key outputs:"
echo "   $SHARED/01_download/z500_evolution_5panel.png"
echo "   $WU/10_fig8/fig8_replica.png"
echo "   $WU/11_xsection/closure_check.png"
echo "   $WU/11_xsection/upper_psi_by_piece.png"
echo "   $WU/11_xsection/xsection_vinduced.png"
echo "   $WU/11_xsection/closure_diagnostic.png"
echo "============================================"
