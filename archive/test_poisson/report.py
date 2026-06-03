#!/usr/bin/env python
"""Generate summary tables and diagnostic plots for test_poisson benchmarks.

Usage:
    micromamba run -n blocking python test_poisson/report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_OUT = Path(__file__).resolve().parent / "outputs"
_OUT.mkdir(exist_ok=True)


def run_all_tests() -> dict:
    """Run the full test suite and capture results."""
    import subprocess

    test_dir = Path(__file__).resolve().parent
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            str(test_dir),
            "-v",
            "--tb=short",
            "-s",  # show print() output for metrics
            "--no-header",
        ],
        capture_output=True, text=True, cwd=str(test_dir.parent),
        timeout=600,  # 10 min max
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def parse_metrics(stdout: str) -> list[dict]:
    """Extract per-solver metrics from test print() output."""
    rows = []
    for line in stdout.splitlines():
        line = line.strip()
        # Look for lines like: "  wu_sor       L²=1.23e-04  pole_err=..."
        if "L²=" in line or "residual=" in line:
            parts = line.split()
            row = {}
            for part in parts:
                if "=" in part:
                    key, val = part.split("=", 1)
                    try:
                        row[key] = float(val.rstrip("s"))
                    except ValueError:
                        row[key] = val
            name = parts[0] if parts else "unknown"
            row["solver"] = name
            if len(row) > 1:
                rows.append(row)
    return rows


def generate_comparison_plots():
    """Generate Wu-vs-Spectral INLIN=1 comparison plots using ERA5 data."""
    import matplotlib
    matplotlib.use("Agg")

    from test_poisson.data_loader import load_wu_event, get_wu_lat_lon
    from test_poisson.test_nonlinear_balnc import (
        run_wu_balnc_inlin1,
        run_backend_balnc_inlin1,
        SpectralBackend,
        make_comparison_plots,
    )
    from wu_python.core.pv_calc import (
        compute_ertel_pv_wu,
        invert_vorticity_balanced,
        compute_relative_vorticity,
    )
    from wu_python.core.nondim import BB, BH, BL

    print("\n" + "=" * 60)
    print("  INLIN=1 Comparison Plots (ERA5 event day)")
    print("=" * 60)

    H, TH, U, V = load_wu_event()
    lat, lon = get_wu_lat_lon()

    print("  Computing Ertel PV...")
    pv = compute_ertel_pv_wu(U, V, TH)
    nlev = pv.shape[0]

    print("  Computing initial balanced ψ...")
    vor = compute_relative_vorticity(U, V)
    psi_init = invert_vorticity_balanced(vor, H, U, V, omega=1.75, max_iter=50000)

    c_k = np.array(BB[:nlev], dtype=np.float64)
    bh_k = np.array(BH[:nlev], dtype=np.float64)
    bl_k = np.array(BL[:nlev], dtype=np.float64)

    H_init = H[:nlev].copy()
    psi_init = psi_init[:nlev].copy()

    print("  Running Wu SOR BALNC (INLIN=1)...")
    wu_result = run_wu_balnc_inlin1(pv, H_init.copy(), psi_init.copy())
    print(f"    converged={wu_result['converged']}, n_outer={wu_result['n_outer']}, "
          f"dt={wu_result['wall_time']:.1f}s")

    print("  Running Spectral SH BALNC (INLIN=1)...")
    sh_result = run_backend_balnc_inlin1(
        pv, H_init.copy(), psi_init.copy(),
        lat, lon, SpectralBackend(), c_k, bh_k, bl_k,
        label="Spectral SH (INLIN=1)",
    )
    print(f"    converged={sh_result['converged']}, n_outer={sh_result['n_outer']}, "
          f"dt={sh_result['wall_time']:.1f}s")

    # Interior agreement
    sl = slice(5, -5)
    psi_wu = wu_result["psi"][:, sl, sl]
    psi_sh = sh_result["psi"][:, sl, sl]
    l2_psi = float(np.linalg.norm(psi_sh - psi_wu)) / (float(np.linalg.norm(psi_wu)) + 1e-30)
    print(f"  ψ L² (interior) = {l2_psi:.4e}")

    for k_level in [2, 5, 7]:
        path = make_comparison_plots(wu_result, sh_result, lat, lon, k_level=k_level)
        print(f"  Plot: {path}")

    print("  Done.\n")


def main():
    print("=" * 70)
    print("  test_poisson — Poisson/Helmholtz Solver Benchmark Report")
    print("=" * 70)

    # ── Run tests ──
    print("\nRunning benchmarks...\n")
    results = run_all_tests()

    if results["returncode"] != 0:
        print("WARNING: Some tests failed. See stderr below:")
        print(results["stderr"][-2000:])

    # ── Parse metrics ──
    metrics = parse_metrics(results["stdout"])

    if metrics:
        print("\n--- Solver Metrics ---")
        header = f"{'Solver':<16} {'L²':>10} {'pole':>10} {'iters':>8} {'residual':>10}"
        print(header)
        print("-" * len(header))
        for m in metrics:
            solver = m.get("solver", "?")
            l2 = m.get("L²", float("nan"))
            pole = m.get("pole_err", float("nan"))
            iters = m.get("iters", float("nan"))
            res = m.get("residual", m.get("rel_res", float("nan")))
            print(f"{solver:<16} {l2:10.4e} {pole:10.4e} {iters:8.0f} {res:10.4e}")

        # Save metrics JSON
        metrics_file = _OUT / "metrics.json"
        metrics_file.write_text(
            json.dumps(metrics, indent=2, default=str)
        )
        print(f"\nMetrics saved to {metrics_file}")

    # ── Generate comparison plots ──
    try:
        generate_comparison_plots()
    except Exception as e:
        print(f"\nPlot generation skipped: {e}")

    # ── Summary ──
    print(f"\n{'─' * 70}")
    print(f"Return code: {results['returncode']}")
    print(f"Full output:  {_OUT / 'pytest_output.txt'}")
    (_OUT / "pytest_output.txt").write_text(results["stdout"] + "\n" + results["stderr"])
    print(f"\nDone.")


if __name__ == "__main__":
    main()
