"""
weak_scaling.py — MCUPS vs problem size using GPU Global kernel across the grid ladder.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_CFL
from gpu_profiles import GPU_PROFILES, get_grids_for_gpu
from utils import ensure_results_dir, save_benchmark_csv
from solver import run_gpu_global


def run_weak_scaling(gpu_key, output_dir):
    """Run weak scaling analysis.

    Outputs:
        - weak_scaling.csv: [nx, ny, total_cells, wall_time, steps, mcups, efficiency_pct]
        - weak_scaling.pdf: MCUPS vs total_cells
    """
    profile = GPU_PROFILES[gpu_key]
    grids   = get_grids_for_gpu(gpu_key, precision="float32")

    print(f"[weak_scaling] GPU: {profile['name']}")
    print(f"[weak_scaling] Grids: {grids}")

    rows = []
    baseline_mcups = None

    for nx, ny in grids:
        print(f"  [GPU-Global] {nx}×{ny} ...", end=" ", flush=True)
        _history, _x, _y, wt, steps, _perf = run_gpu_global(nx, ny, [2.0], CFL=DEFAULT_CFL)
        cells = nx * ny
        mcups = cells * int(steps) / wt / 1e6

        if baseline_mcups is None:
            baseline_mcups = mcups
        efficiency = mcups / baseline_mcups * 100.0

        rows.append([nx, ny, cells, f"{wt:.4f}", int(steps), f"{mcups:.2f}", f"{efficiency:.1f}"])
        print(f"{wt:.2f}s  {mcups:.1f} MCUPS  eff={efficiency:.1f}%")

    # ---- Save CSV ----
    csv_path = os.path.join(output_dir, "weak_scaling.csv")
    save_benchmark_csv(
        csv_path,
        ["nx", "ny", "total_cells", "wall_time", "steps", "mcups", "efficiency_pct"],
        rows,
    )

    # ---- Plot ----
    _plot_weak_scaling(rows, profile, output_dir)
    print(f"[weak_scaling] Done.")


def _plot_weak_scaling(rows, profile, output_dir):
    cells      = [int(r[2])   for r in rows]
    mcups_vals = [float(r[5]) for r in rows]
    effs       = [float(r[6]) for r in rows]

    peak = max(mcups_vals)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(cells, mcups_vals, marker="o", linestyle="-", label="GPU Global")
    ax.axhline(peak, color="gray", linestyle="--", linewidth=0.9, label=f"Peak {peak:.0f} MCUPS")

    for c, m, e in zip(cells, mcups_vals, effs):
        ax.annotate(f"{e:.0f}%", xy=(c, m), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8)

    ax.set_xlabel("Total cells (N×M)")
    ax.set_ylabel("MCUPS")
    ax.set_title(f"Weak Scaling — {profile['short_name']}")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()

    pdf_path = os.path.join(output_dir, "weak_scaling.pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[SAVED] {pdf_path}")
