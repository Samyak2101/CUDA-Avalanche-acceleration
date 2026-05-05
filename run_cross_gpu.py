"""Cross-GPU comparison — generates comparative analysis plots."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gpu_profiles import GPU_PROFILES
from utils import ensure_results_dir

OUTPUT_DIR = ensure_results_dir("cross_gpu")
GPUS_TO_COMPARE = ["gtx1650", "a5000", "rtx5070", "v100"]
GPU_COLORS = {"gtx1650": "#e74c3c", "a5000": "#3498db", "rtx5070": "#2ecc71", "v100": "#9b59b6"}
GPU_MARKERS = {"gtx1650": "o", "a5000": "s", "rtx5070": "^", "v100": "D"}
RESULTS_BASE = os.path.join(os.path.dirname(__file__), "results")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def load_csv(gpu_key, filename):
    """Load a CSV from a GPU results directory. Returns list-of-dicts or None."""
    path = os.path.join(RESULTS_BASE, gpu_key, filename)
    if not os.path.exists(path):
        print(f"  [WARN] Missing: {path} — skipping {gpu_key}")
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def available_gpus(filename):
    """Return list of GPU keys that have the given CSV file."""
    return [g for g in GPUS_TO_COMPARE
            if os.path.exists(os.path.join(RESULTS_BASE, g, filename))]


# ---------------------------------------------------------------------------
# Plot 1: MCUPS Comparison (weak scaling)
# ---------------------------------------------------------------------------
def plot_mcups_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for gpu in GPUS_TO_COMPARE:
        rows = load_csv(gpu, "weak_scaling.csv")
        if rows is None:
            continue
        try:
            cells = [float(r["total_cells"]) for r in rows]
            mcups = [float(r["mcups"]) for r in rows]
        except KeyError:
            print(f"  [WARN] weak_scaling.csv for {gpu} missing expected columns")
            continue
        label = GPU_PROFILES[gpu]["name"]
        ax.plot(cells, mcups, color=GPU_COLORS[gpu], marker=GPU_MARKERS[gpu],
                label=label, linewidth=2, markersize=7)
        plotted += 1

    if plotted == 0:
        print("[SKIP] No weak_scaling data found — skipping Plot 1")
        plt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_xlabel("Total Cells")
    ax.set_ylabel("Throughput (MCUPS)")
    ax.set_title("Throughput Comparison Across GPUs")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cross_gpu_mcups.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: Bandwidth Utilization (at 800×260)
# ---------------------------------------------------------------------------
def plot_bandwidth_utilization():
    REFERENCE_CELLS = 800 * 260  # 208 000

    gpus, utils_pct, measured_bws = [], [], []
    for gpu in GPUS_TO_COMPARE:
        rows = load_csv(gpu, "weak_scaling.csv")
        if rows is None:
            continue
        peak_bw = GPU_PROFILES[gpu]["mem_bw_gbs"]
        # Find row closest to reference cell count
        try:
            best = min(rows, key=lambda r: abs(float(r["total_cells"]) - REFERENCE_CELLS))
            bw = float(best.get("eff_bw_gbs", best.get("bandwidth_gbs", 0)))
        except (KeyError, ValueError):
            continue
        gpus.append(GPU_PROFILES[gpu]["name"])
        measured_bws.append(bw)
        utils_pct.append(bw / peak_bw * 100.0)

    if not gpus:
        print("[SKIP] No bandwidth data found — skipping Plot 2")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(gpus))
    bars = ax.bar(x, utils_pct, color=[GPU_COLORS[g] for g in GPUS_TO_COMPARE if GPU_PROFILES[g]["name"] in gpus],
                  width=0.55, edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars, utils_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(gpus, rotation=15, ha="right")
    ax.set_ylabel("Bandwidth Utilization (%)")
    ax.set_title("Memory Bandwidth Utilization at 800×260 Grid")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cross_gpu_bandwidth.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3: FP64 Penalty
# ---------------------------------------------------------------------------
def plot_fp64_penalty():
    gpus, ratios = [], []
    for gpu in GPUS_TO_COMPARE:
        rows = load_csv(gpu, "precision_study.csv")
        if rows is None:
            continue
        fp32, fp64 = None, None
        for r in rows:
            prec = r.get("precision", "").upper()
            if prec == "FP32":
                fp32 = float(r.get("mcups", 0))
            elif prec == "FP64":
                fp64 = float(r.get("mcups", 0))
        if fp32 and fp64 and fp64 > 0:
            gpus.append(GPU_PROFILES[gpu]["name"])
            ratios.append(fp32 / fp64)

    if not gpus:
        print("[SKIP] No precision_study data found — skipping Plot 3")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(gpus))
    colors = [GPU_COLORS[g] for g in GPUS_TO_COMPARE if GPU_PROFILES[g]["name"] in gpus]
    bars = ax.bar(x, ratios, color=colors, width=0.55, edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}×", ha="center", va="bottom", fontsize=9)
    ax.axhline(2, linestyle="--", color="gray", linewidth=1, label="Ideal (1:2)")
    ax.set_xticks(x)
    ax.set_xticklabels(gpus, rotation=15, ha="right")
    ax.set_ylabel("FP32 / FP64 MCUPS Ratio")
    ax.set_title("FP64 Performance Penalty by Architecture")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cross_gpu_fp64_penalty.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 4: Combined Roofline
# ---------------------------------------------------------------------------
def plot_combined_roofline():
    # Arithmetic intensity range for flux kernel (typical: 0.1–2.0 FLOP/byte)
    ai_range = np.logspace(-2, 2, 500)
    flux_ai = 0.25  # representative AI for flux kernel

    fig, ax = plt.subplots(figsize=(9, 6))
    for gpu in GPUS_TO_COMPARE:
        p = GPU_PROFILES[gpu]
        name = p["name"]
        bw = p["mem_bw_gbs"]       # GB/s
        fp32 = p["fp32_tflops"] * 1e3  # GFLOP/s

        # Memory ceiling: perf = AI * BW
        mem_line = ai_range * bw
        # Compute ceiling: horizontal at fp32
        roof = np.minimum(mem_line, fp32)

        ax.plot(ai_range, roof, color=GPU_COLORS[gpu], linewidth=2, label=name)

        # Flux kernel operating point: estimate MCUPS from weak_scaling if available
        rows = load_csv(gpu, "weak_scaling.csv")
        if rows:
            try:
                best = max(rows, key=lambda r: float(r["total_cells"]))
                measured_gflops = float(best.get("eff_bw_gbs", bw * 0.4)) * flux_ai
                ax.scatter([flux_ai], [measured_gflops], color=GPU_COLORS[gpu],
                           marker=GPU_MARKERS[gpu], s=80, zorder=5)
            except (KeyError, ValueError):
                pass

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title("Comparative Roofline Model")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.annotate("Flux kernel\noperating region", xy=(flux_ai, 1),
                xytext=(flux_ai * 3, 1), fontsize=9, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray"))
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cross_gpu_roofline.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 5: Strong Scaling Speedup
# ---------------------------------------------------------------------------
def plot_strong_scaling_speedup():
    # Collect per-GPU speedup at common grid sizes
    data = {}  # gpu -> {grid_label -> speedup}
    for gpu in GPUS_TO_COMPARE:
        rows = load_csv(gpu, "strong_scaling.csv")
        if rows is None:
            continue
        entry = {}
        for r in rows:
            try:
                grid = r.get("grid", r.get("grid_size", ""))
                speedup = float(r.get("speedup_vs_cpu", r.get("speedup", 0)))
                entry[grid] = speedup
            except (KeyError, ValueError):
                continue
        if entry:
            data[gpu] = entry

    if not data:
        print("[SKIP] No strong_scaling data found — skipping Plot 5")
        return

    # Common grids present in at least one GPU
    all_grids = sorted({g for v in data.values() for g in v})
    x = np.arange(len(all_grids))
    n_gpus = len(data)
    width = 0.7 / max(n_gpus, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (gpu, entry) in enumerate(data.items()):
        heights = [entry.get(g, 0) for g in all_grids]
        offset = (i - n_gpus / 2 + 0.5) * width
        ax.bar(x + offset, heights, width=width, label=GPU_PROFILES[gpu]["name"],
               color=GPU_COLORS[gpu], edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(all_grids, rotation=20, ha="right")
    ax.set_ylabel("Speedup vs CPU (Numba JIT)")
    ax.set_title("GPU Speedup Over CPU (Numba JIT)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cross_gpu_speedup.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------
def write_summary_csv():
    REFERENCE_CELLS = 800 * 260

    rows_out = []
    for gpu in GPUS_TO_COMPARE:
        p = GPU_PROFILES[gpu]
        row = {
            "GPU": p["name"],
            "Peak_BW_GBs": p["mem_bw_gbs"],
            "Peak_FP32_TFLOPS": p["fp32_tflops"],
            "Measured_BW_GBs": "",
            "BW_Utilization_Pct": "",
            "Peak_MCUPS": "",
            "Speedup_vs_CPU_800x260": "",
        }
        ws_rows = load_csv(gpu, "weak_scaling.csv")
        if ws_rows:
            try:
                best = min(ws_rows, key=lambda r: abs(float(r["total_cells"]) - REFERENCE_CELLS))
                bw = float(best.get("eff_bw_gbs", best.get("bandwidth_gbs", 0)))
                mcups = float(best.get("mcups", 0))
                row["Measured_BW_GBs"] = f"{bw:.2f}"
                row["BW_Utilization_Pct"] = f"{bw / p['mem_bw_gbs'] * 100:.1f}"
                row["Peak_MCUPS"] = f"{mcups:.1f}"
            except (KeyError, ValueError):
                pass

        ss_rows = load_csv(gpu, "strong_scaling.csv")
        if ss_rows:
            try:
                target = "800x260"
                match = next((r for r in ss_rows
                              if r.get("grid", r.get("grid_size", "")) == target), None)
                if match:
                    row["Speedup_vs_CPU_800x260"] = match.get("speedup_vs_cpu", match.get("speedup", ""))
            except (KeyError, StopIteration):
                pass

        rows_out.append(row)

    if not any(r["Measured_BW_GBs"] for r in rows_out):
        print("[SKIP] No data for summary CSV")
        return

    out = os.path.join(OUTPUT_DIR, "cross_gpu_summary.csv")
    fieldnames = list(rows_out[0].keys())
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gpus_present = available_gpus("weak_scaling.csv")
    if len(gpus_present) < 2:
        print(f"[WARN] Only {len(gpus_present)} GPU result(s) found. "
              "Run at least 2 GPU runners first for meaningful comparison.")
    else:
        print(f"Comparing GPUs: {gpus_present}")

    print("\n--- Plot 1: MCUPS Comparison ---")
    plot_mcups_comparison()

    print("\n--- Plot 2: Bandwidth Utilization ---")
    plot_bandwidth_utilization()

    print("\n--- Plot 3: FP64 Penalty ---")
    plot_fp64_penalty()

    print("\n--- Plot 4: Combined Roofline ---")
    plot_combined_roofline()

    print("\n--- Plot 5: Strong Scaling Speedup ---")
    plot_strong_scaling_speedup()

    print("\n--- Summary CSV ---")
    write_summary_csv()

    print(f"\nDone. Outputs in: {OUTPUT_DIR}")
