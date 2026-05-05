"""
strong_scaling.py — Fixed problem-size wall-time comparison: CPU Numba vs GPU Global vs GPU Shared.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_CFL
from gpu_profiles import GPU_PROFILES, get_grids_for_gpu
from utils import timer, ensure_results_dir, save_benchmark_csv
from solver import run_cpu_numba, run_gpu_global, run_gpu_shared

CPU_MAX_GRID = (800, 260)   # cap CPU runs to avoid excessive runtime


def _run_solver(solver_fn, nx, ny, t_end=2.0, cfl=DEFAULT_CFL):
    """Run a solver and return (wall_time_s, steps).

    All solvers share the signature:
        solver_fn(nx, ny, target_times, CFL=cfl)
    and return at least (history, x, y, wall_time, steps[, perf]).
    """
    result = solver_fn(nx, ny, [t_end], CFL=cfl)
    # result is a 5-tuple (CPU) or 6-tuple (GPU); wall_time at index 3, steps at 4
    wall_time = result[3]
    steps     = result[4]
    return wall_time, int(steps)


def run_strong_scaling(gpu_key, output_dir):
    """Run strong scaling analysis.

    Outputs:
        - strong_scaling.csv: [nx, ny, total_cells, solver, wall_time, steps, mcups, speedup_vs_cpu]
        - strong_scaling.pdf: log-scale wall_time vs total_cells per solver
    """
    profile = GPU_PROFILES[gpu_key]
    grids   = get_grids_for_gpu(gpu_key, precision="float32")

    print(f"[strong_scaling] GPU: {profile['name']}")
    print(f"[strong_scaling] Grids: {grids}")

    rows = []
    cpu_times = {}   # (nx,ny) -> wall_time  (only for grids <= CPU_MAX_GRID)

    # ---- CPU Numba (capped) ----
    cpu_grids = [(nx, ny) for nx, ny in grids
                 if nx <= CPU_MAX_GRID[0] and ny <= CPU_MAX_GRID[1]]
    for nx, ny in cpu_grids:
        print(f"  [CPU ] {nx}×{ny} ...", end=" ", flush=True)
        wt, steps = _run_solver(run_cpu_numba, nx, ny)
        cells = nx * ny
        mcups = cells * steps / wt / 1e6
        cpu_times[(nx, ny)] = wt
        rows.append([nx, ny, cells, "cpu_numba", f"{wt:.4f}", steps, f"{mcups:.2f}", "1.000"])
        print(f"{wt:.2f}s  {mcups:.1f} MCUPS")

    # ---- GPU Global ----
    for nx, ny in grids:
        print(f"  [GPU-Global] {nx}×{ny} ...", end=" ", flush=True)
        wt, steps = _run_solver(run_gpu_global, nx, ny)
        cells = nx * ny
        mcups = cells * steps / wt / 1e6
        cpu_ref = cpu_times.get((nx, ny))
        speedup = f"{cpu_ref/wt:.2f}" if cpu_ref else "N/A"
        rows.append([nx, ny, cells, "gpu_global", f"{wt:.4f}", steps, f"{mcups:.2f}", speedup])
        print(f"{wt:.2f}s  {mcups:.1f} MCUPS  speedup={speedup}")

    # ---- GPU Shared ----
    for nx, ny in grids:
        print(f"  [GPU-Shared] {nx}×{ny} ...", end=" ", flush=True)
        wt, steps = _run_solver(run_gpu_shared, nx, ny)
        cells = nx * ny
        mcups = cells * steps / wt / 1e6
        cpu_ref = cpu_times.get((nx, ny))
        speedup = f"{cpu_ref/wt:.2f}" if cpu_ref else "N/A"
        rows.append([nx, ny, cells, "gpu_shared", f"{wt:.4f}", steps, f"{mcups:.2f}", speedup])
        print(f"{wt:.2f}s  {mcups:.1f} MCUPS  speedup={speedup}")

    # ---- Save CSV ----
    csv_path = os.path.join(output_dir, "strong_scaling.csv")
    save_benchmark_csv(
        csv_path,
        ["nx", "ny", "total_cells", "solver", "wall_time", "steps", "mcups", "speedup_vs_cpu"],
        rows,
    )

    # ---- Plot ----
    _plot_strong_scaling(rows, profile, output_dir)
    print(f"[strong_scaling] Done.")


def _plot_strong_scaling(rows, profile, output_dir):
    import csv

    data = {}
    for r in rows:
        nx, ny, cells, solver = int(r[0]), int(r[1]), int(r[2]), r[3]
        wt = float(r[4])
        data.setdefault(solver, {"cells": [], "wt": []})
        data[solver]["cells"].append(cells)
        data[solver]["wt"].append(wt)

    styles = {
        "cpu_numba":  dict(marker="o", linestyle="--", label="CPU Numba"),
        "gpu_global": dict(marker="s", linestyle="-",  label="GPU Global"),
        "gpu_shared": dict(marker="^", linestyle="-",  label="GPU Shared"),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    for solver, d in data.items():
        s = styles.get(solver, {})
        ax.loglog(d["cells"], d["wt"], **s)

    ax.set_xlabel("Total cells (N×M)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(f"Strong Scaling — {profile['short_name']}")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()

    pdf_path = os.path.join(output_dir, "strong_scaling.pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[SAVED] {pdf_path}")
