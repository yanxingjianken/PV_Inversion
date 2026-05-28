#!/bin/bash
# 04_run_wu.sh
# Run the Wu/Davis piecewise PV inversion pipeline for the Jan 2025 CA blocking event.
#
# Pipeline:
#   Pass A: pvpialln on clim .grid  → meanq, meanh (climatological mean state PV/balanced psi)
#   Pass B: pvpialln on event .grid → event_q.out, event_h.out (event total PV/balanced psi)
#   Pass C: qinvert21 on event      → event_bal.out (refined balanced streamfunction)
#   Pass D: qinvertp21 (3 pieces)   → event_pert.out (piecewise perturbation streamfunction)
#
# Working directory: case_jan2025_CA/wu_inv/
# Executables:       inv3d_ca2025/  (relative to pv_inversion/)
#
# Grid:  NX=87, NY=51, NW=10
# Levels: 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200 hPa
# Pieces:
#   Piece 1 (lower):  levels 1-2  (1000, 925 hPa)
#   Piece 2 (middle): levels 3-4  (850, 700 hPa)
#   Piece 3 (upper):  levels 5-9  (600, 500, 400, 300, 250 hPa)

set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WU_DIR="${CASE_DIR}/wu_inv"
EXE_DIR="/net/flood/data2/users/x_yan/pv_inversion/inv3d_ca2025"

echo "=== Wu PV Inversion Pipeline ==="
echo "CASE_DIR  : ${CASE_DIR}"
echo "WU_DIR    : ${WU_DIR}"
echo "EXE_DIR   : ${EXE_DIR}"
echo ""

# --------------------------------------------------------------------------
# Verify executables
# --------------------------------------------------------------------------
for exe in pvpialln.exe qinvert21.exe qinvertp21.exe; do
    if [[ ! -x "${EXE_DIR}/${exe}" ]]; then
        echo "ERROR: ${EXE_DIR}/${exe} not found or not executable"
        exit 1
    fi
done
echo "All executables found."

# --------------------------------------------------------------------------
# Verify input .grid files
# --------------------------------------------------------------------------
CLIM_GRID="${WU_DIR}/mean_jan08_00z.grid"
EVENT_GRID="${WU_DIR}/20250108_00.grid"
for f in "${CLIM_GRID}" "${EVENT_GRID}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: $f not found. Run 03_write_grid.py first."
        exit 1
    fi
done
echo "Input .grid files found."
echo ""

# --------------------------------------------------------------------------
# PASS A: pvpialln on clim .grid → meanq, meanh
# (single input = clim state → mean state = clim)
# --------------------------------------------------------------------------
echo "=== Pass A: pvpialln (clim) → meanq, meanh ==="
cd "${WU_DIR}"

# Remove old output files (pvpialln uses STATUS='new' — fails if files exist)
rm -f meanq meanh clim_jan08_q.out clim_jan08_h.out

"${EXE_DIR}/pvpialln.exe" << 'EOF'
meanq
meanh
1
mean_jan08_00z.grid
1
1
clim_jan08_q.out
clim_jan08_h.out
EOF
echo "  → meanq, meanh, clim_jan08_q.out, clim_jan08_h.out"

if [[ ! -f meanq || ! -f meanh ]]; then
    echo "ERROR: Pass A failed — meanq or meanh not produced."
    exit 1
fi
echo "Pass A complete."
echo ""

# --------------------------------------------------------------------------
# PASS B: pvpialln on event .grid → event_q.out, event_h.out
# (single input = event → event total PV and balanced psi)
# Note: the "mean" files from this pass (event_meanq_discard, event_meanh_discard)
#       are NOT used; they represent the mean of the event-only state.
# --------------------------------------------------------------------------
echo "=== Pass B: pvpialln (event) → event_q.out, event_h.out ==="
cd "${WU_DIR}"

rm -f event_meanq_discard event_meanh_discard event_q.out event_h.out

"${EXE_DIR}/pvpialln.exe" << 'EOF'
event_meanq_discard
event_meanh_discard
1
20250108_00.grid
1
1
event_q.out
event_h.out
EOF
echo "  → event_q.out, event_h.out"

if [[ ! -f event_q.out || ! -f event_h.out ]]; then
    echo "ERROR: Pass B failed — event_q.out or event_h.out not produced."
    exit 1
fi
echo "Pass B complete."
echo ""

# --------------------------------------------------------------------------
# PASS C: qinvert21 on event → event_bal.out
# Refines the balanced streamfunction via SOR on the full PV field.
# Input:  event_h.out (geopotential height + streamfunction)
#         event_q.out (boundary θ + interior PV)
# Output: event_bal.out (refined balanced height + streamfunction)
# --------------------------------------------------------------------------
echo "=== Pass C: qinvert21 (event) → event_bal.out ==="
cd "${WU_DIR}"

rm -f event_bal.out

"${EXE_DIR}/qinvert21.exe" << 'EOF'
200
200
1.85
1.75
0.5
0.1
'event_h.out'
'event_q.out'
'event_bal.out'
1
0.01
1
EOF
echo "  → event_bal.out"

if [[ ! -f event_bal.out ]]; then
    echo "ERROR: Pass C failed — event_bal.out not produced."
    exit 1
fi
echo "Pass C complete."
echo ""

# --------------------------------------------------------------------------
# PASS D: qinvertp21 (piecewise) → event_pert.out
# Inverts the perturbation PV in 3 vertical layers.
# Input:  meanq          (clim mean PV + boundary θ)
#         meanh          (clim mean height + streamfunction)
#         event_q.out    (event total PV + boundary θ)
#         event_bal.out  (event balanced height + streamfunction)
# Output: event_pert.out (piecewise perturbation streamfunction for 3 pieces)
#
# Pieces:
#   1: levels 1-2  (1000, 925 hPa) — lower troposphere
#   2: levels 3-4  (850, 700 hPa)  — middle troposphere
#   3: levels 5-9  (600-250 hPa)   — upper troposphere
#
# Output/inversion levels: all 10 (1..10)
# INLIN=1 (nonlinear balance terms included)
# IQD=0   (no external PV-dependency file)
# IBC=0   (homogeneous BCs for all pieces)
# --------------------------------------------------------------------------
echo "=== Pass D: qinvertp21 (3 pieces) → event_pert.out ==="
cd "${WU_DIR}"

rm -f event_pert.out

"${EXE_DIR}/qinvertp21.exe" << 'EOF'
1.4
1.4
0.5
0.01
1.
1.
'meanq'
'meanh'
'event_q.out'
'event_bal.out'
'event_pert.out'
1
0
0
10,1,2,3,4,5,6,7,8,9,10
10,1,2,3,4,5,6,7,8,9,10
3
2,1,2
0
2,3,4
0
5,5,6,7,8,9
0
EOF
echo "  → event_pert.out"

if [[ ! -f event_pert.out ]]; then
    echo "ERROR: Pass D failed — event_pert.out not produced."
    exit 1
fi
echo "Pass D complete."
echo ""

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo "=== Pipeline complete ==="
echo "Output files:"
ls -lh "${WU_DIR}/event_pert.out" \
        "${WU_DIR}/event_bal.out" \
        "${WU_DIR}/event_q.out" \
        "${WU_DIR}/event_h.out" \
        "${WU_DIR}/meanq" \
        "${WU_DIR}/meanh"
echo ""
echo "Next step: run 06_read_outs.py to convert to NetCDF."
