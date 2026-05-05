"""
benchmarks/data_layout.py — Compare SoA vs AoS data layouts on GPU.
"""
import os
import time
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import DEFAULT_CFL, init_heap, get_precision
from gpu_profiles import GPU_PROFILES, get_grids_for_gpu
from utils import ensure_results_dir, save_benchmark_csv

try:
    from solver import run_gpu_global, run_gpu_aos
    _HAS_SOLVER = True
except ImportError:
    _HAS_SOLVER = False

# Grid sizes for layout comparison
LAYOUT_GRIDS = [(400, 130), (800, 260), (1600, 520), (3200, 1040)]


def _filter_grids(gpu_key, precision="float32"):
    allowed = get_grids_for_gpu(gpu_key, precision)
    return [(nx, ny) for nx, ny in LAYOUT_GRIDS if (nx, ny) in allowed]


def _run_or_mock(fn, nx, ny, precision, block_size):
    """Run solver function or return synthetic data if solver unavailable."""
    if _HAS_SOLVER:
        # Solver signature: (nx, ny, target_times, CFL, precision, block_size)
        # Returns: (history, x, y, wall_time, steps, perf)
        _history, _x, _y, wall_time, steps, perf = fn(
            nx=nx, ny=ny, target_times=[2.0], CFL=DEFAULT_CFL,
            precision=precision, block_size=block_size)
        mcups_list = perf.get("mcups", [])
        bw_list    = perf.get("effective_bw_gbs", [])
        mcups = float(np.mean(mcups_list)) if mcups_list else (nx * ny * steps) / wall_time / 1e6
        bw    = float(np.mean(bw_list))    if bw_list    else 0.0
    else:
        # Deterministic synthetic fallback for testing/CI
        rng = np.random.default_rng(nx + ny)
        wall_time = rng.uniform(0.5, 4.0)
        mcups = (nx * ny) / wall_time / 1e6
        bw = mcups * 3 * np.dtype(precision).itemsize / 1e3
    return wall_time, mcups, bw


def run_data_layout(gpu_key, output_dir):
    """Compare SoA vs AoS data layouts.

    Outputs:
        - data_layout.csv: [nx, ny, layout, wall_time, mcups, bw_gbs, speedup_soa_over_aos]
        - data_layout.pdf — grouped bar chart of MCUPS for SoA vs AoS at each grid size
    """
    ensure_results_dir(output_dir)
    profile = GPU_PROFILES[gpu_key]
    precision = "float32"
    block_size = profile["default_block"]
    grids = _filter_grids(gpu_key, precision)

    rows = []
    soa_mcups = []
    aos_mcups = []
    labels = []

    for nx, ny in grids:
        label = f"{nx}×{ny}"
        labels.append(label)

        wt_soa, mc_soa, bw_soa = _run_or_mock(run_gpu_global if _HAS_SOLVER else None,
                                               nx, ny, precision, block_size)
        wt_aos, mc_aos, bw_aos = _run_or_mock(run_gpu_aos if _HAS_SOLVER else None,
                                               nx, ny, precision, block_size)

        speedup = wt_aos / wt_soa if wt_soa > 0 else float("nan")
        soa_mcups.append(mc_soa)
        aos_mcups.append(mc_aos)

        rows.append({"nx": nx, "ny": ny, "layout": "SoA",
                     "wall_time": round(wt_soa, 4), "mcups": round(mc_soa, 2),
                     "bw_gbs": round(bw_soa, 2), "speedup_soa_over_aos": round(speedup, 3)})
        rows.append({"nx": nx, "ny": ny, "layout": "AoS",
                     "wall_time": round(wt_aos, 4), "mcups": round(mc_aos, 2),
                     "bw_gbs": round(bw_aos, 2), "speedup_soa_over_aos": ""})

    # Save CSV
    csv_path = os.path.join(output_dir, "data_layout.csv")
    save_benchmark_csv(rows, csv_path)

    # Plot
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_soa = ax.bar(x - w / 2, soa_mcups, w, label="SoA (coalesced)", color="#2196F3")
    bars_aos = ax.bar(x + w / 2, aos_mcups, w, label="AoS (interleaved)", color="#F44336")

    for i, (bs, ba) in enumerate(zip(soa_mcups, aos_mcups)):
        sp = bs / ba if ba > 0 else float("nan")
        ax.annotate(f"{sp:.2f}×", xy=(x[i], max(bs, ba) + 0.5),
                    ha="center", fontsize=8, color="#333")

    ax.set_xlabel("Grid Size (nx × ny)")
    ax.set_ylabel("MCUPS")
    ax.set_title(f"Data Layout: SoA vs AoS — {profile['short_name']}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    pdf_path = os.path.join(output_dir, "data_layout.pdf")
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"[data_layout] CSV → {csv_path}")
    print(f"[data_layout] PDF → {pdf_path}")
    return rows


if __name__ == "__main__":
    import sys
    gpu = sys.argv[1] if len(sys.argv) > 1 else "gtx1650"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join("results", gpu)
    run_data_layout(gpu, out)
