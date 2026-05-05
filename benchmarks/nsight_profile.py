"""
benchmarks/nsight_profile.py — Nsight Compute integration for the Savage-Hutter solver.

Generates:
  - benchmarks/_nsight_snippet.py   : minimal solver run for ncu to attach to
  - results/<gpu>/nsight/nsight_commands.bat : ready-to-run ncu command lines
  - results/<gpu>/nsight/nsight_results.csv  : parsed metrics (if ncu is on PATH)
"""

import os
import csv
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpu_profiles import GPU_PROFILES
from utils import ensure_results_dir

# ---------------------------------------------------------------------------
# Kernels to profile and their CUDA kernel function names
# ---------------------------------------------------------------------------

KERNELS = [
    "compute_k_grid_kernel",
    "compute_wave_speeds_kernel",
    "compute_fluxes_global_kernel",
    "update_state_kernel",
]

# Targeted sections — fast and sufficient for roofline / memory-bound analysis.
# SpeedOfLight   : achieved vs. theoretical BW & FLOP ceilings (1 pass).
# MemoryWorkload  : L1 / L2 / DRAM sector breakdown (1–2 passes).
# Avoids --set full which replays the kernel dozens of times.
NCU_SECTIONS = [
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
]

# Extra individual metrics not covered by the sections above.
NCU_METRICS = [
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.sum",
]

_SNIPPET_NAME = "_nsight_snippet.py"
_SNIPPET_CONTENT = '''\
# -*- coding: utf-8 -*-
"""Minimal solver run for Nsight Compute profiling -- auto-generated."""
import sys
sys.path.insert(0, r'd:\\cuda')
from solver.gpu_global import run_gpu_global
# 10 timesteps is enough for ncu to capture individual kernel launches
run_gpu_global(800, 260, [0.01] * 10, CFL=0.3, precision='float32')
'''


# ---------------------------------------------------------------------------
# Helper: write the snippet script
# ---------------------------------------------------------------------------

def _write_snippet(benchmarks_dir: str) -> str:
    path = os.path.join(benchmarks_dir, _SNIPPET_NAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_SNIPPET_CONTENT)
    return path


# ---------------------------------------------------------------------------
# Helper: build ncu command lines
# ---------------------------------------------------------------------------

def _ncu_commands(snippet_path: str, output_dir: str) -> list[str]:
    sections_flags = " ".join(f"--section {s}" for s in NCU_SECTIONS)
    metrics_str    = ",".join(NCU_METRICS)
    cmds = []
    for kernel in KERNELS:
        csv_out = os.path.join(output_dir, f"ncu_{kernel}.csv")
        rep_out = os.path.join(output_dir, f"ncu_{kernel}")  # .ncu-rep auto-appended
        cmd = (
            f'ncu'
            f' {sections_flags}'
            f' --metrics {metrics_str}'
            f' --kernel-name {kernel}'
            f' --launch-skip 2'       # skip JIT / warm-up launches
            f' --launch-count 3'      # profile 3 steady-state launches
            f' --target-processes all'
            f' --csv'
            f' --log-file "{csv_out}"'
            f' --export "{rep_out}"'  # .ncu-rep for Nsight Compute GUI
            f' python "{snippet_path}"'
        )
        cmds.append(cmd)
    return cmds


# ---------------------------------------------------------------------------
# Helper: try to parse ncu CSV output
# ---------------------------------------------------------------------------

def _parse_ncu_csv(csv_path: str) -> list[dict]:
    """Extract metric rows from an ncu --csv output file."""
    rows = []
    if not os.path.isfile(csv_path):
        return rows
    with open(csv_path, newline="", errors="replace") as f:
        # ncu csv has some header comment lines starting with "=="
        lines = [l for l in f if not l.startswith("==")]
    reader = csv.DictReader(lines)
    for row in reader:
        rows.append(dict(row))
    return rows


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_nsight_profile(gpu_key: str = "gtx1650", output_dir: str = None):
    """
    Generate Nsight Compute profiling commands and optionally run them.

    Parameters
    ----------
    gpu_key    : key in GPU_PROFILES
    output_dir : directory for outputs; defaults to results/<gpu_key>/nsight/

    Outputs
    -------
    _nsight_snippet.py    — standalone profiling script
    nsight_commands.bat   — runnable ncu command lines
    nsight_results.csv    — parsed metrics per kernel (only if ncu ran)
    """
    profile = GPU_PROFILES[gpu_key]
    benchmarks_dir = os.path.dirname(os.path.abspath(__file__))

    if output_dir is None:
        output_dir = os.path.join(
            benchmarks_dir, "..", "results", gpu_key, "nsight"
        )
    ensure_results_dir(output_dir)

    # ---- snippet script -----------------------------------------------------
    snippet_path = _write_snippet(benchmarks_dir)
    print(f"[nsight] Snippet written -> {snippet_path}")

    # ---- command file -------------------------------------------------------
    cmds = _ncu_commands(snippet_path, output_dir)
    bat_path = os.path.join(output_dir, "nsight_commands.bat")
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write("@echo off\n")
        f.write(f"REM Nsight Compute profiling -- {profile['short_name']}\n")
        f.write("REM Run this file from the d:\\cuda directory as Administrator.\n\n")
        for cmd in cmds:
            f.write(cmd + "\n\n")
    print(f"[nsight] Command file saved -> {bat_path}")

    # ---- try running ncu if available ---------------------------------------
    ncu_exe = shutil.which("ncu") or shutil.which("nv-nsight-cu-cli")
    if ncu_exe is None:
        print("[nsight] ncu not found on PATH — skipping live profiling.")
        print("         Run nsight_commands.bat manually after installing Nsight Compute.")
        return {"bat_path": bat_path, "ncu_ran": False}

    print(f"[nsight] ncu found at {ncu_exe} — running profiling ...")
    all_rows = []
    for kernel, cmd in zip(KERNELS, cmds):
        print(f"[nsight]   profiling {kernel} ...")
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=120
            )
            csv_out = os.path.join(output_dir, f"ncu_{kernel}.csv")
            parsed  = _parse_ncu_csv(csv_out)
            for row in parsed:
                row["kernel"] = kernel
            all_rows.extend(parsed)
            if result.returncode != 0:
                print(f"[nsight]   WARNING: ncu returned {result.returncode} for {kernel}")
        except subprocess.TimeoutExpired:
            print(f"[nsight]   TIMEOUT for {kernel} — skipping.")
        except Exception as e:
            print(f"[nsight]   ERROR for {kernel}: {e}")

    # ---- save combined results CSV ------------------------------------------
    if all_rows:
        combined_csv = os.path.join(output_dir, "nsight_results.csv")
        fieldnames = list(all_rows[0].keys())
        with open(combined_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[nsight] Results saved -> {combined_csv}")
    else:
        print("[nsight] No parsed results — check individual ncu_*.csv files.")

    return {"bat_path": bat_path, "ncu_ran": True, "rows": all_rows}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Nsight Compute profiling integration")
    p.add_argument("--gpu", default="gtx1650",
                   choices=list(GPU_PROFILES.keys()),
                   help="GPU profile key")
    p.add_argument("--out", default=None, help="Output directory")
    args = p.parse_args()
    run_nsight_profile(gpu_key=args.gpu, output_dir=args.out)
