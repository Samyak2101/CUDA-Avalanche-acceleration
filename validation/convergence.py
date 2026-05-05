"""
convergence.py — Spatial and temporal convergence study.
Run from d:\\cuda\\:  python -m validation.convergence
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from math import log

from utils import nearest_time_index, binned_profile, fill_nans_linear, ensure_results_dir
from solver import run_gpu_global

# ── Constants ─────────────────────────────────────────────────────────────────
CFL         = 0.3
TARGET_TIME = 1.0
Y_TARGET    = 0.65
NBINS       = 800
X_MIN, X_MAX = 0.0, 4.0

SPATIAL_GRIDS = [(100, 32), (200, 65), (400, 130), (800, 260)]
CFL_VALUES    = [0.8, 0.6, 0.4, 0.2, 0.1]
FIXED_GRID    = (400, 130)

# ── Load TITAN2D reference ────────────────────────────────────────────────────
def load_titan_reference():
    npz_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "titan_height_cells_0to2s.npz")
    data    = np.load(npz_path)
    t_titan = data['t']
    xc      = data['xc']
    yc      = data['yc']
    H       = data['H']

    tidx = nearest_time_index(t_titan, TARGET_TIME)
    sel  = (np.abs(yc - Y_TARGET) <= 0.002) & (xc >= X_MIN) & (xc <= X_MAX)
    xr   = xc[sel]
    hr   = H[tidx, sel]
    mids, hb = binned_profile(xr, hr, X_MIN, X_MAX, NBINS)
    hb = fill_nans_linear(mids, hb)
    return mids, hb


def extract_centerline(h_field, nx, ny, x_coords, y_coords):
    """Return (x_1d, h_centerline) along y ≈ Y_TARGET."""
    jc  = int(np.argmin(np.abs(y_coords - Y_TARGET)))
    return x_coords, h_field[jc, :]


# ── Spatial convergence ───────────────────────────────────────────────────────
def spatial_convergence(ref_x, ref_h):
    from config import Lx
    errors, dxs = [], []
    print("\n=== Spatial Convergence ===")
    for (nx, ny) in SPATIAL_GRIDS:
        dx = Lx / (nx - 1)
        print(f"  Grid {nx}x{ny}  dx={dx:.5f} ...", flush=True)
        history, x, y, wt, steps, _ = run_gpu_global(nx, ny, [TARGET_TIME], CFL=CFL)
        h = history[TARGET_TIME]
        xc_num, hc_num = extract_centerline(h, nx, ny, x, y)
        # interpolate onto ref grid
        h_interp = np.interp(ref_x, xc_num, hc_num)
        rmse = np.sqrt(np.mean((h_interp - ref_h) ** 2))
        errors.append(rmse)
        dxs.append(dx)
        print(f"    RMSE={rmse:.6f}  wall={wt:.1f}s  steps={steps}")

    # Order of accuracy
    print("\n  Order of accuracy (spatial):")
    for i in range(len(errors) - 1):
        p = log(errors[i] / errors[i+1]) / log(2.0)
        print(f"    {SPATIAL_GRIDS[i]} -> {SPATIAL_GRIDS[i+1]}:  p = {p:.3f}")

    return np.array(dxs), np.array(errors)


# ── Temporal convergence ──────────────────────────────────────────────────────
def temporal_convergence(ref_x, ref_h):
    from config import Lx, Ly
    nx, ny = FIXED_GRID
    dx = Lx / (nx - 1)
    errors, avg_dts = [], []
    print("\n=== Temporal Convergence ===")
    for cfl in CFL_VALUES:
        print(f"  CFL={cfl} ...", flush=True)
        history, x, y, wt, steps, _ = run_gpu_global(nx, ny, [TARGET_TIME], CFL=cfl)
        h = history[TARGET_TIME]
        xc_num, hc_num = extract_centerline(h, nx, ny, x, y)
        h_interp = np.interp(ref_x, xc_num, hc_num)
        rmse = np.sqrt(np.mean((h_interp - ref_h) ** 2))
        avg_dt = TARGET_TIME / max(steps, 1)
        errors.append(rmse)
        avg_dts.append(avg_dt)
        print(f"    RMSE={rmse:.6f}  avg_dt={avg_dt:.5f}  wall={wt:.1f}s  steps={steps}")

    print("\n  Order of accuracy (temporal):")
    for i in range(len(errors) - 1):
        if errors[i] > 0 and errors[i+1] > 0:
            p = log(errors[i] / errors[i+1]) / log(avg_dts[i] / avg_dts[i+1])
            print(f"    CFL {CFL_VALUES[i]} -> {CFL_VALUES[i+1]}:  p = {p:.3f}")

    return np.array(avg_dts), np.array(errors)


# ── Plot ──────────────────────────────────────────────────────────────────────
def make_plot(dxs, sp_err, dts, tm_err):
    out_dir = ensure_results_dir("validation")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Spatial
    ax = axes[0]
    ax.loglog(dxs, sp_err, 'o-', label='RMSE vs Δx')
    ref = sp_err[0] * (dxs / dxs[0])
    ax.loglog(dxs, ref, 'k--', label='O(Δx)')
    ax.set_xlabel('Δx (m)')
    ax.set_ylabel('RMSE (m)')
    ax.set_title('Spatial Convergence')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    # Temporal
    ax = axes[1]
    ax.loglog(dts, tm_err, 's-', color='tab:orange', label='RMSE vs Δt')
    ref = tm_err[0] * (dts / dts[0])
    ax.loglog(dts, ref, 'k--', label='O(Δt)')
    ax.set_xlabel('avg Δt (s)')
    ax.set_ylabel('RMSE (m)')
    ax.set_title('Temporal Convergence')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    out = os.path.join(out_dir, "convergence.pdf")
    fig.savefig(out)
    print(f"\n[SAVED] {out}")
    plt.close(fig)


# ── Public entry point (called by GPU runner scripts) ─────────────────────────
def run_convergence():
    """Run the full spatial + temporal convergence study and save plots."""
    print("Loading TITAN2D reference...")
    ref_x, ref_h = load_titan_reference()

    dxs,  sp_err = spatial_convergence(ref_x, ref_h)
    dts,  tm_err = temporal_convergence(ref_x, ref_h)
    make_plot(dxs, sp_err, dts, tm_err)
    print("Convergence study done.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading TITAN2D reference...")
    ref_x, ref_h = load_titan_reference()

    dxs,  sp_err = spatial_convergence(ref_x, ref_h)
    dts,  tm_err = temporal_convergence(ref_x, ref_h)
    make_plot(dxs, sp_err, dts, tm_err)
    print("Done.")
