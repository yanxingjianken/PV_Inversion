#!/usr/bin/env python3
"""Clean ALL intermediate files, executables, NC outputs, and plots from wu/ pipeline.

KEEPS: data/wu_in/{mean,event}.grid (cached shared_steps output)
       shared_steps/ data (ERA5 downloads, clim netCDFs)
       wu_python/ (untouched per user directive)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # pv_inversion/
DATA = ROOT / "data"
WU_IN = DATA / "wu_in"
WU_OUT = DATA / "wu_out"
BUILD = DATA / "wu_bin"
FIGS = DATA / "figs"
WU_STEPS = ROOT / "wu" / "steps"

# 1. Fortran executables
print("Cleaning executables...")
for pattern in ["*.exe", "*.o", "*.mod"]:
    for f in BUILD.glob(pattern):
        print(f"  rm {f}")
        f.unlink()

# 2. Wu intermediate ASCII outputs (keep .grid files!)
print("Cleaning wu_in/ intermediates...")
keep = {"mean.grid", "event.grid"}
for f in sorted(WU_IN.iterdir()):
    if f.name not in keep and not f.is_dir():
        print(f"  rm {f}")
        f.unlink()

# 3. Wu output NetCDFs
print("Cleaning wu_out/...")
for f in WU_OUT.glob("*.nc"):
    print(f"  rm {f}")
    f.unlink()

# 4. Figures
print("Cleaning figs/...")
for pattern in ["*.png", "*.pdf"]:
    for f in FIGS.glob(pattern):
        print(f"  rm {f}")
        f.unlink()

# 5. Step-generated plots
print("Cleaning step plots...")
for step_dir in sorted(WU_STEPS.iterdir()):
    if not step_dir.is_dir():
        continue
    for pattern in ["*.png", "*.pdf"]:
        for f in step_dir.glob(pattern):
            print(f"  rm {f}")
            f.unlink()

# 6. Any __pycache__
print("Cleaning __pycache__...")
for pyc in ROOT.rglob("__pycache__"):
    for f in pyc.iterdir():
        f.unlink()

print("\n✓ Clean complete. Ready for regeneration.")
print(f"  Kept: {', '.join(sorted(keep))}")
