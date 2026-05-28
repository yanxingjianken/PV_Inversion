#!/bin/bash
# 04_run_ppvi.sh
# ==============
# Full PPVI pipeline for the 2025-01-08 00Z California blocking case.
#
# Environment split:
#   - 'blocking'      env → ppvi_engine.py  (needs pyspharm/pvtend)
#   - 'fourcastnetv2' env → all other Python scripts
#
# Usage (from the case root directory):
#   bash scripts/04_run_ppvi.sh [--skip-download] [--skip-clim] [--skip-ppvi]
#
# Steps:
#   01  Download ERA5 NH event data            (fourcastnetv2)
#   02  Prepare NH climatological basic state  (fourcastnetv2)
#   03  [SKIP – Wu grid writer no longer used]
#   04  Run PPVI inversion engine              (blocking)
#   06  Compute PV advection diagnostics        (fourcastnetv2)
#   08  Render 2×2 Fig-8 replica figures        (fourcastnetv2)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CASE_DIR"

SKIP_DOWNLOAD=0
SKIP_CLIM=0
SKIP_PPVI=0

for arg in "$@"; do
  case "$arg" in
    --skip-download) SKIP_DOWNLOAD=1 ;;
    --skip-clim)     SKIP_CLIM=1     ;;
    --skip-ppvi)     SKIP_PPVI=1     ;;
  esac
done

echo "================================================================"
echo " PPVI pipeline — 2025-01-08 00Z CA blocking"
echo " Case dir: $CASE_DIR"
echo "================================================================"

# ── Step 01: Download ERA5 event (NH, 8 levels) ───────────────────────────
if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
  echo ""
  echo "[01] Downloading ERA5 NH event data (fourcastnetv2) …"
  micromamba run -n fourcastnetv2 python scripts/01_download_era5.py
else
  echo "[01] Skipping ERA5 download (--skip-download)"
fi

# ── Step 02: Prepare NH climatology ──────────────────────────────────────
if [[ $SKIP_CLIM -eq 0 ]]; then
  echo ""
  echo "[02] Preparing NH climatological basic state (fourcastnetv2) …"
  micromamba run -n fourcastnetv2 python scripts/02_prep_clim.py
else
  echo "[02] Skipping clim prep (--skip-clim)"
fi

# ── Step 04: PPVI inversion (blocking env — needs pyspharm) ──────────────
if [[ $SKIP_PPVI -eq 0 ]]; then
  echo ""
  echo "[04] Running PPVI inversion engine (blocking env with pyspharm) …"
  micromamba run -n blocking python scripts/ppvi_engine.py
else
  echo "[04] Skipping PPVI inversion (--skip-ppvi)"
fi

# ── Step 06: Compute PV advection diagnostics ─────────────────────────────
echo ""
echo "[06] Computing PV advection (fourcastnetv2) …"
micromamba run -n blocking python scripts/06_read_outs.py

# ── Step 08: Plot 2×2 Fig-8 replica ──────────────────────────────────────
echo ""
echo "[08] Rendering 2×2 figures (fourcastnetv2) …"
micromamba run -n fourcastnetv2 python scripts/08_plot_2x2.py

echo ""
echo "================================================================"
echo " Pipeline complete."
echo " Figures in: $CASE_DIR/figs/"
echo "================================================================"
